# src/train/sft2.py

import torch
import random
import math
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image

from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import ACTION_TOKENS
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..tools.visual_utils import draw_cursor
from ..utils.sft_screenspot import ScreenSpotDataManager, get_shortest_path_actions

# =============================================================================
# 1. Logic Helpers (Diverse Oracle CoT Generator)
# =============================================================================

# --- Template Library ---
COT_TEMPLATES = {
    # 1. Intent / Goal
    "intent": [
        "The user wants to '{instruction}'.",
        "Goal: Execute the command '{instruction}'.",
        "Instruction received: '{instruction}'.",
    ],
    
    # 2. Localization (Cursor & Target)
    "localization": [
        "The cursor is currently in the **{c_region}** region, while the target lies in the **{t_region}** region.",
        "Scanning screen... Cursor found at **{c_region}**. Target identified at **{t_region}**.",
        "Position check: Cursor is at **{c_region}**; Target is at **{t_region}**.",
    ],
    
    # 3. Relative Direction (Movement needed)
    "direction": [
        "The target is located to the **{rel_pos}** of the current cursor position.",
        "To reach the target, I need to move towards the **{rel_pos}**.",
        "There is a spatial offset. The target is **{rel_pos}** relative to the cursor.",
    ],
    
    # 4. Arrival (Click needed)
    "arrival": [
        "The cursor is positioned **over** the target.",
        "Target acquired. The cursor is aligned with the element.",
        "Zero distance. The cursor is exactly where it needs to be.",
    ],
    
    # 5. Action Justification (Move - Far)
    "plan_move_far": [
        "There is a significant gap. I need a large jump.",
        "The distance is large. A long-range movement is required.",
        "Far away. I will execute a coarse movement to close the gap.",
    ],
    
    # 6. Action Justification (Move - Mid)
    "plan_move_mid": [
        "The target is moderately away. I need a standard step.",
        "Medium distance detected. A normal move command is appropriate.",
        "Closing the distance. I will move closer.",
    ],
    
    # 7. Action Justification (Move - Close)
    "plan_move_close": [
        "The target is very close. I need a micro-adjustment.",
        "Almost there. A fine-tuning step is needed.",
        "Short distance. I will nudge the cursor slightly.",
    ],
    
    # 8. Action Justification (Click)
    "plan_click": [
        "I will perform a click.",
        "Executing click action.",
        "Interaction required. Clicking now.",
    ]
}

def get_grid_region(x: int, y: int, width: int, height: int) -> str:
    """Determine 3x3 grid region."""
    if y < height / 3: v_tag = "Top"
    elif y < 2 * height / 3: v_tag = "Mid"
    else: v_tag = "Bottom"
    
    if x < width / 3: h_tag = "Left"
    elif x < 2 * width / 3: h_tag = "Center"
    else: h_tag = "Right"
    
    if v_tag == "Mid" and h_tag == "Center": return "Center"
    elif v_tag == "Mid": return f"Mid-{h_tag}"
    else: return f"{v_tag}-{h_tag}"

def generate_cot_for_step(cursor_pos, target_pos, instruction, next_action_token):
    """
    Generate 'Oracle' Chain-of-Thought with High Entropy + Precise Coords.
    """
    cx, cy = cursor_pos
    tx, ty = target_pos
    
    # --- 1. Select Templates (Random Mix & Match) ---
    t_intent = random.choice(COT_TEMPLATES["intent"])
    t_loc = random.choice(COT_TEMPLATES["localization"])
    
    # --- 2. Fill Data ---
    c_region = get_grid_region(cx, cy, HP.IMAGE_SIZE, HP.IMAGE_SIZE)
    t_region = get_grid_region(tx, ty, HP.IMAGE_SIZE, HP.IMAGE_SIZE)
    
    cot_part1 = f"Reasoning: {t_intent.format(instruction=instruction)} "
    cot_part2 = t_loc.format(c_region=c_region, t_region=t_region)
    
    cot = f"{cot_part1}{cot_part2} "
    
    # --- [NEW] Exact Relative Coordinates ---
    rel_cx, rel_cy = round(cx / HP.IMAGE_SIZE, 1), round(cy / HP.IMAGE_SIZE, 1)
    rel_tx, rel_ty = round(tx / HP.IMAGE_SIZE, 1), round(ty / HP.IMAGE_SIZE, 1)
    
    if c_region == t_region:
        cot += f"I need to examine the position more carefully. "
        cot += f"The cursor is at about [{rel_cx:.1f}, {rel_cy:.1f}] and the target is at about [{rel_tx:.1f}, {rel_ty:.1f}] (relative coordinates) of the image. "

    # --- 3. Spatial Logic ---
    dx = tx - cx
    dy = ty - cy
    margin = 15
    
    v_rel, h_rel = "", ""
    if dy < -margin: v_rel = "Top"
    elif dy > margin: v_rel = "Bottom"
    if dx < -margin: h_rel = "Left"
    elif dx > margin: h_rel = "Right"
    
    rel_pos_str = ""
    if v_rel and h_rel: rel_pos_str = f"{v_rel}-{h_rel}"
    elif v_rel: rel_pos_str = v_rel
    elif h_rel: rel_pos_str = h_rel
    else: rel_pos_str = "Overlapping"

    # --- 4. Movement vs Arrival ---
    if rel_pos_str != "Overlapping":
        t_dir = random.choice(COT_TEMPLATES["direction"])
        cot += t_dir.format(rel_pos=rel_pos_str) + " "
    else:
        t_arr = random.choice(COT_TEMPLATES["arrival"])
        cot += t_arr + " "

    # --- [NEW] Direction Prioritization (Only for Move) ---
    if "MOVE" in next_action_token:
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        cot += f"Currently the **{direction}** direction is the farthest away. "

    # --- 5. Action Planning ---
    if "CLICK" in next_action_token:
        t_plan = random.choice(COT_TEMPLATES["plan_click"])
        cot += t_plan
    else:
        if "FAR" in next_action_token:
            t_plan = random.choice(COT_TEMPLATES["plan_move_far"])
        elif "MID" in next_action_token:
            t_plan = random.choice(COT_TEMPLATES["plan_move_mid"])
        else:
            t_plan = random.choice(COT_TEMPLATES["plan_move_close"])
        cot += t_plan

    return f"{cot}\nAction: {next_action_token}"

# =============================================================================
# 2. Weighted Trainer (Action Emphasis)
# =============================================================================

class WeightedActionTrainer(Trainer):
    """
    Applies 100x weight to the Action Token.
    Smartly ignores <|im_end|> and Newlines.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Robust Tokenizer Access
        self.my_tokenizer = self.tokenizer if self.tokenizer else getattr(self, "processing_class", None)
        if hasattr(self.my_tokenizer, "tokenizer"):
            self.my_tokenizer = self.my_tokenizer.tokenizer
            
        # 1. Pre-compute Valid Action IDs
        self.valid_action_ids = set()
        from ..utils.action import ACTION_TOKENS 
        for t in ACTION_TOKENS:
            ids = self.my_tokenizer.encode(t, add_special_tokens=False)
            if ids: self.valid_action_ids.add(ids[-1])
            ids_space = self.my_tokenizer.encode(" " + t, add_special_tokens=False)
            if ids_space: self.valid_action_ids.add(ids_space[-1])
        
        # 2. Blacklist IDs
        self.im_end_id = self.my_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_id = self.my_tokenizer.eos_token_id
        self.newline_id = 198 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        batch_size = labels.size(0)
        vocab_size = shift_logits.size(-1)
        
        raw_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        raw_loss = raw_loss.view(batch_size, -1)

        weights = torch.ones_like(raw_loss)
        ACTION_WEIGHT = 100.0 

        for i in range(batch_size):
            valid_mask = (shift_labels[i] != -100)
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
            
            if len(valid_indices) > 0:
                target_idx = None
                
                # Smart Backtracking
                search_window = torch.flip(valid_indices, [0])[:15]
                
                # Pass 1: Strict Match
                for idx in search_window:
                    token_id = shift_labels[i, idx].item()
                    if token_id in self.valid_action_ids:
                        target_idx = idx
                        break
                
                # Pass 2: Fallback (Skip Blacklist)
                if target_idx is None:
                    for idx in search_window:
                        token_id = shift_labels[i, idx].item()
                        if token_id not in [self.im_end_id, self.eos_id, self.newline_id]:
                            target_idx = idx
                            break
                
                if target_idx is not None:
                    weights[i, target_idx] = ACTION_WEIGHT

        valid_tokens_mask = (shift_labels != -100).float()
        final_weights = weights * valid_tokens_mask
        
        loss = (raw_loss * final_weights).sum() / (final_weights.sum() + 1e-8)
        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 3. Dataset & Collator
# =============================================================================

@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images = [], []
        for feature in features:
            msgs = feature["messages"]
            formatted_msgs = [
                {
                    "role": "user", 
                    "content": [{"type": "image"}, {"type": "text", "text": msgs[1]["content"]}]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": msgs[2]["content"]}]
                }
            ]
            text = self.processor.apply_chat_template(formatted_msgs, tokenize=False, add_generation_prompt=False)
            text = f"{AGENT_SYSTEM_PROMPT}\n{text}"
            texts.append(text)
            images.append(feature["image"])
            
        batch = self.processor(
            text=texts, images=images, 
            padding=True, truncation=True, max_length=2048, return_tensors="pt"
        )
        
        # Mask User Prompt
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i in range(len(input_ids)):
            start_indices = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
            if len(start_indices) >= 2:
                last_turn_idx = start_indices[-1]
                labels[i, :last_turn_idx + 1] = -100 
        
        batch["labels"] = labels
        return batch

class ScreenSpotSFTDataset(Dataset):
    def __init__(self, split="train"):
        self.ss_manager = ScreenSpotDataManager()
        self.split = split
        if split == "train": self.data_source = self.ss_manager.raw_train
        else: self.data_source = self.ss_manager.raw_eval
        
        print(f"[SFT-2 Dataset] Split '{split}': {len(self.data_source)} base samples.")

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return self._process_sample(self.data_source[idx])

    def _process_sample(self, sample):
        # 1. Prepare Image
        raw_img = sample['image'].convert("RGB")
        orig_w, orig_h = raw_img.size
        image = raw_img.resize((HP.IMAGE_SIZE, HP.IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        # 2. Scale Target BBox (Denormalize on the fly)
        bbox = sample.get('bbox', None)
        if not bbox and 'point' in sample:
            p = sample['point']
            bbox = [p[0], p[1], p[0], p[1]]
        
        if bbox:
            # Check if normalized (0.0 - 1.0)
            if all(0.0 <= c <= 1.0 for c in bbox):
                abs_x1 = bbox[0] * orig_w
                abs_y1 = bbox[1] * orig_h
                abs_x2 = bbox[2] * orig_w
                abs_y2 = bbox[3] * orig_h
            else:
                abs_x1, abs_y1, abs_x2, abs_y2 = bbox

            # Scale to HP.IMAGE_SIZE
            scale_x = HP.IMAGE_SIZE / orig_w
            scale_y = HP.IMAGE_SIZE / orig_h
            
            tx = int(((abs_x1 + abs_x2) / 2) * scale_x)
            ty = int(((abs_y1 + abs_y2) / 2) * scale_y)
            
            # Clamp
            tx = max(0, min(HP.IMAGE_SIZE-1, tx))
            ty = max(0, min(HP.IMAGE_SIZE-1, ty))
        else:
            tx, ty = HP.IMAGE_SIZE // 2, HP.IMAGE_SIZE // 2

        # 3. Generate Trajectory Path
        center_x, center_y = HP.IMAGE_SIZE // 2, HP.IMAGE_SIZE // 2
        path = get_shortest_path_actions((center_x, center_y), (tx, ty))
        
        # 4. Sampling Strategy
        if not path:
            current_pos = (center_x, center_y)
            action_token = "<CLICK_SHORT>"
        else:
            # 33% chance to force Final Step (Click)
            if random.random() < 0.33:
                step_idx = len(path) - 1
            else:
                # 67% chance to teach Navigation
                if len(path) > 1:
                    step_idx = random.randint(0, len(path) - 2)
                else:
                    step_idx = 0
            
            target_action = path[step_idx]
            action_token = target_action[0]
            
            # Simulate state
            curr_cx, curr_cy = center_x, center_y
            for i in range(step_idx):
                _, (nx, ny) = path[i]
                curr_cx, curr_cy = nx, ny
            current_pos = (curr_cx, curr_cy)

        # 5. Draw Cursor
        image = draw_cursor(image, current_pos[0], current_pos[1])
        
        # 6. Generate Label
        cot_response = generate_cot_for_step(current_pos, (tx, ty), sample['instruction'], action_token)
        
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"[Action] {sample['instruction']}"},
            {"role": "assistant", "content": cot_response}
        ]
        
        return {"messages": msgs, "image": image}

# =============================================================================
# 4. Run Logic
# =============================================================================

def run_sft_screenspot():
    input_path = HP.SFT_2_INPUT_PATH
    output_path = HP.SFT_2_OUTPUT_PATH
    
    if not os.path.exists(input_path):
        print(f"[Error] Stage 1 model not found at {input_path}")
        return

    print(f"[SFT-2] Loading Stage 1 Model from {input_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        input_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(input_path, trust_remote_code=True)
    
    # Freeze Vision, Train LLM+Embeds
    model.enable_input_require_grads()
    model.get_input_embeddings().weight.requires_grad = True
    for name, param in model.named_parameters():
        if "visual" in name: param.requires_grad = False
        else: param.requires_grad = True
        
    train_dataset = ScreenSpotSFTDataset(split="train")
    eval_dataset = ScreenSpotSFTDataset(split="eval")
    collator = SFTDataCollator(processor=processor)
    
    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=HP.SFT_2_EPOCHS,
        per_device_train_batch_size=HP.SFT_2_BATCH_SIZE,
        gradient_accumulation_steps=HP.SFT_2_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_2_LEARN_RATE,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_safetensors=True,
        eval_strategy="steps",
        save_steps=120,
        eval_steps=120,
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = WeightedActionTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator, processing_class=processor
    )
    
    print("[SFT-2] Starting Weighted Trajectory Training (Action Weight = 100x)...")
    trainer.train()
    
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    train_dataset.ss_manager.save_test_set()

if __name__ == "__main__":
    run_sft_screenspot()