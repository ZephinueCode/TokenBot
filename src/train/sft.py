# src/train/sft.py

import json
import torch
import os
import random
import math
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import ACTION_TOKENS
from ..utils.prompts import AGENT_SYSTEM_PROMPT

# =============================================================================
# 0. Custom Trainer for Action Reweighting
# =============================================================================

class WeightedActionTrainer(Trainer):
    """
    Applies 30x weight to the Action Token.
    Crucially ignores <|im_end|> and Newlines to focus on the Action.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. Pre-compute Valid Action Token IDs
        self.valid_action_ids = set()
        if self.processing_class:
            for t in ACTION_TOKENS:
                # Encode "TOKEN"
                ids = self.processing_class.tokenizer.encode(t, add_special_tokens=False)
                if ids: self.valid_action_ids.add(ids[-1])
                # Encode " TOKEN" (with space)
                ids_space = self.processing_class.tokenizer.encode(" " + t, add_special_tokens=False)
                if ids_space: self.valid_action_ids.add(ids_space[-1])
        
        # 2. Identify Blacklist IDs
        self.im_end_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_id = self.processing_class.tokenizer.eos_token_id
        self.newline_id = 198 # ID for '\n' in Qwen/Llama

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift for Causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        batch_size = labels.size(0)
        vocab_size = shift_logits.size(-1)
        
        raw_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        raw_loss = raw_loss.view(batch_size, -1)

        weights = torch.ones_like(raw_loss)
        ACTION_WEIGHT = 10.0

        for i in range(batch_size):
            valid_mask = (shift_labels[i] != -100)
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
            
            if len(valid_indices) > 0:
                target_idx = None
                
                # [CRITICAL LOGIC] Backtrack to find Action Token
                # Search last 15 tokens. Priority: Action Token > Non-Special
                search_window = torch.flip(valid_indices, [0])[:15]
                
                # Pass 1: Strict match for Action Token ID
                for idx in search_window:
                    token_id = shift_labels[i, idx].item()
                    if token_id in self.valid_action_ids:
                        target_idx = idx
                        break
                
                # Pass 2: Fallback (if tokenization mismatch), skip blacklist
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
# 1. Semantic Grid & Logic (UPDATED CoT)
# =============================================================================

def get_grid_region(x: int, y: int, width: int, height: int) -> str:
    if y < height / 3: v_tag = "Top"
    elif y < 2 * height / 3: v_tag = "Mid"
    else: v_tag = "Bottom"
    if x < width / 3: h_tag = "Left"
    elif x < 2 * width / 3: h_tag = "Center"
    else: h_tag = "Right"
    
    if v_tag == "Mid" and h_tag == "Center": return "Center"
    elif v_tag == "Mid": return f"Mid-{h_tag}"
    else: return f"{v_tag}-{h_tag}"

def get_spatial_label_with_cot(cursor_pos, target_center, intent):
    """
    Generates reasoning + Special Token Action.
    Now includes relative coordinates and directional prioritization logic.
    """
    # === Navigation Logic (No Coordinates needed usually) ===
    if intent["type"] == "nav":
        mode = intent["mode"] # 'back' or 'home'
        reasoning = "Reasoning: "
        
        if mode == "back":
            reasoning += "I need to go back to the last page. "
            reasoning += "I do not need to interact with a specific element on the screen. "
            reasoning += "I need to return to the previous state. "
            action = "<GO_BACK>"
        elif mode == "home":
            reasoning += "The instruction requires returning to the main interface. "
            reasoning += "I need to exit the current application context. "
            action = "<GO_HOME>"
        
        return f"{reasoning}\nAction: {action}"
    
    # === Interaction Logic ===
    cx, cy = cursor_pos
    tx, ty = target_center
    
    dx = tx - cx
    dy = ty - cy
    dist = math.hypot(dx, dy)
    
    c_region = get_grid_region(cx, cy, HP.IMAGE_SIZE, HP.IMAGE_SIZE)
    t_region = get_grid_region(tx, ty, HP.IMAGE_SIZE, HP.IMAGE_SIZE)
    target_name = intent.get('name', 'target')
    
    # 1. Basic Region Description
    reasoning = "Reasoning: "
    reasoning += f"The cursor is currently in the **{c_region}** region. "
    reasoning += f"The target '{target_name}' is located in the **{t_region}** region. "

    # 2. [NEW] Relative Coordinates
    # Calculate relative position (0.0 - 1.0)
    rel_cx, rel_cy = round(cx / HP.IMAGE_SIZE, 1), round(cy / HP.IMAGE_SIZE, 1)
    rel_tx, rel_ty = round(tx / HP.IMAGE_SIZE, 1), round(ty / HP.IMAGE_SIZE, 1)
    
    # 3. Relative Position Logic
    if c_region == t_region:
        reasoning += f"I need to examine the position more carefully. "
        reasoning += f"The cursor is at about [{rel_cx:.1f}, {rel_cy:.1f}] and the target is at about [{rel_tx:.1f}, {rel_ty:.1f}] (relative coordinates) of the image. "
    
    margin = 15
    v_rel = ""
    h_rel = ""

    if dy < -margin: v_rel = "Top"
    elif dy > margin: v_rel = "Bottom"
    
    if dx < -margin: h_rel = "Left"
    elif dx > margin: h_rel = "Right"
    
    rel_pos_str = ""
    if v_rel and h_rel: rel_pos_str = f"{v_rel}-{h_rel}"
    elif v_rel: rel_pos_str = v_rel
    elif h_rel: rel_pos_str = h_rel
    else: rel_pos_str = "Overlapping"

    if rel_pos_str != "Overlapping":
        reasoning += f"The target is to the **{rel_pos_str}** of the cursor. "
    else:
        reasoning += f"The cursor is right on the target. "
    
    HIT_THRESHOLD = 15
    action = ""
    
    # === Decision: Click vs Move ===
    if dist < HIT_THRESHOLD:
        reasoning += f"The cursor is currently positioned **over** the target '{target_name}'. "
        if intent["type"] == "click":
            reasoning += "I will perform a click."
            action = "<CLICK_SHORT>"
        elif intent["type"] == "long_click":
            reasoning += "I will perform a long press."
            action = "<CLICK_LONG>"
        elif intent["type"] == "text":
            content = intent.get("content", "text")
            reasoning += f"I need to type input."
            action = f"<TEXT_START> {content} <TEXT_END>"
        else:
            reasoning += "Task complete."
            action = "<END_ACTION>"
    else:
        # Describe Relative Direction
        dir_desc = []
        if abs(dx) > abs(dy):
            if dx > 0: dir_desc.append("to the right")
            else: dir_desc.append("to the left")
        else:
            if dy > 0: dir_desc.append("downwards")
            else: dir_desc.append("upwards")
            
        reasoning += f"The target is {' '.join(dir_desc)} of the cursor. "
        
        # 4. [NEW] Direction Prioritization Logic
        # Compare horizontal vs vertical gap to decide main direction
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
            reasoning += f"Currently the **{direction}** direction is the farthest away. "
            
            d_val = abs(dx)
            suffix = "FAR" if d_val > 300 else ("MID" if d_val > 100 else "CLO")
            
            if suffix == "FAR": reasoning += "There is a significant gap. I need a large jump. "
            elif suffix == "MID": reasoning += "The target is moderately away. I need a standard step. "
            else: reasoning += "The target is very close. I need a micro-adjustment. "

            action = f"<MOVE_{direction}_{suffix}>"
        else:
            direction = "DOWN" if dy > 0 else "UP"
            reasoning += f"Currently the **{direction}** direction is the farthest away. "
            
            d_val = abs(dy)
            suffix = "FAR" if d_val > 300 else ("MID" if d_val > 100 else "CLO")
            
            if suffix == "FAR": reasoning += "There is a significant gap. I need a large jump. "
            elif suffix == "MID": reasoning += "The target is moderately away. I need a standard step. "
            else: reasoning += "The target is very close. I need a micro-adjustment. "
            
            action = f"<MOVE_{direction}_{suffix}>"
            
    return f"{reasoning}\nAction: {action}"

# =============================================================================
# 2. Dataset
# =============================================================================
class SFTDataset(Dataset):
    def __init__(self, data_path):
        # 1. Load Semantic Data
        if not os.path.exists(data_path):
            print(f"[Error] {data_path} not found.")
            self.jsonl_data = []
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.jsonl_data = [json.loads(line) for line in f]
        
        # 2. Load Backgrounds
        self.bg_images = self._load_real_images(limit=200)
        
        self.num_spatial = len(self.jsonl_data)
        self.total_len = len(self.jsonl_data) + self.num_spatial
        
        print(f"[Dataset] Total Samples: {self.total_len}")
        
        self.ui_names = ["Submit", "Cancel", "Search", "Menu", "Settings", "Back", "Profile", "Login", "Options", "Button", "Application", "Game", "Function"]
        self.text_contents = ["hello", "test", "123456", "user", "admin"]
        
        self.prompts_click = [
            "Click {name}", "Select {name}", "Tap on {name}", "Hit the {name} button", "Choose {name}"
        ]
        self.prompts_move = [
            "Move to {name}", "Hover over {name}", "Point at {name}", "Go to {name}", "Find {name}"
        ]
        self.prompts_text = [
            "Type '{content}'", "Enter '{content}' ", "Fill {name} with '{content}'", "Input '{content}'"
        ]
        self.prompts_back = [
            "Go back", "Return to previous page", "Back", "Navigate back", "Backspace", "Backwards"
        ]
        self.prompts_home = [
            "Go home", "Return to main screen", "Home", "Exit to desktop", "Quit", "Homepage"
        ]

    def _load_real_images(self, limit=200):
        images = []
        try:
            ds = load_dataset("rootsautomation/ScreenSpot", split="test", streaming=True)
            iterator = iter(ds)
            for _ in range(limit):
                try:
                    sample = next(iterator)
                    img = sample['image'].convert("RGB").resize((HP.IMAGE_SIZE, HP.IMAGE_SIZE))
                    images.append(img)
                except StopIteration:
                    break
        except Exception as e:
            pass
        return images

    def get_random_background(self):
        if self.bg_images:
            return random.choice(self.bg_images).copy()
        return Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # === TYPE 1: SEMANTIC (JSONL) ===
        if idx < len(self.jsonl_data):
            item = self.jsonl_data[idx]
            data_type = item.get("data_type", "action")
            if data_type == "explanation":
                image = Image.new('RGB', (HP.IMAGE_SIZE, HP.IMAGE_SIZE), color=(0, 0, 0))
            else:
                random_pixels = np.random.randint(0, 255, (HP.IMAGE_SIZE, HP.IMAGE_SIZE, 3), dtype=np.uint8)
                image = Image.fromarray(random_pixels)
            msgs = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + item["messages"]
            return {"messages": msgs, "image": image}

        # === TYPE 2: SPATIAL & NAVIGATION (Synthetic) ===
        else:
            image = self.get_random_background()
            draw = ImageDraw.Draw(image)
            margin = 50
            
            sample_rng = random.random()
            
            tx = random.randint(margin, HP.IMAGE_SIZE - margin)
            ty = random.randint(margin, HP.IMAGE_SIZE - margin)
            target_name = random.choice(self.ui_names)
            
            # CASE A: Moving (70%)
            if sample_rng < 0.7:
                while True:
                    cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                    cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                    if math.hypot(cx-tx, cy-ty) > 50: break
                
                sub_type = random.choice(["click", "move", "text"])
                if sub_type == "click":
                    intent = {"type": "click", "name": target_name}
                    instr_tmpl = random.choice(self.prompts_click)
                elif sub_type == "move":
                    intent = {"type": "move", "name": target_name}
                    instr_tmpl = random.choice(self.prompts_move)
                else:
                    content = random.choice(self.text_contents)
                    intent = {"type": "text", "name": target_name, "content": content}
                    instr_tmpl = random.choice(self.prompts_text)
                
                instr = instr_tmpl.format(name=target_name, content=intent.get("content", ""))

            # CASE B: Direct Clicking (15%)
            elif sample_rng < 0.85:
                cx = tx + random.randint(-20, 20)
                cy = ty + random.randint(-20, 20)
                intent = {"type": "click", "name": target_name}
                instr = random.choice(self.prompts_click).format(name=target_name)

            # CASE C: Other Tasks (15%)
            else:
                if random.random() < 0.5:
                    # Text Ready
                    cx = tx + random.randint(-10, 10)
                    cy = ty + random.randint(-10, 10)
                    content = random.choice(self.text_contents)
                    intent = {"type": "text", "name": target_name, "content": content}
                    instr = random.choice(self.prompts_text).format(name=target_name, content=content)
                else:
                    # Navigation
                    intent_mode = random.choice(["back", "home"])
                    intent = {"type": "nav", "mode": intent_mode, "name": "System"}
                    cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                    cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                    tx, ty = 0, 0
                    
                    if intent_mode == "back": instr = random.choice(self.prompts_back)
                    else: instr = random.choice(self.prompts_home)

            # Draw
            if intent.get("type") != "nav":
                draw.rectangle([tx-30, ty-20, tx+30, ty+20], outline=random.choice(("green","black","blue","yellow","red")), width=3)
                try: draw.text((tx-20, ty-35), target_name, fill=random.choice(("green","black","blue","yellow","red")))
                except: pass

            for _ in range(random.randint(2, 5)):
                dx_rand = random.randint(margin, HP.IMAGE_SIZE - margin)
                dy_rand = random.randint(margin, HP.IMAGE_SIZE - margin)
                if abs(dx_rand - tx) > 60 or abs(dy_rand - ty) > 60:
                    draw.rectangle([dx_rand-30, dy_rand-20, dx_rand+30, dy_rand+20], outline=random.choice(("green","black","blue","yellow","red")), width=3)

            from ..tools.visual_utils import draw_cursor
            image = draw_cursor(image, cx, cy)

            cot_response = get_spatial_label_with_cot((cx, cy), (tx, ty), intent)
            
            msgs = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"[Action] {instr}"},
                {"role": "assistant", "content": cot_response}
            ]
            
            return {"messages": msgs, "image": image}

# =============================================================================
# 3. Collator
# =============================================================================
@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for feature in features:
            original_messages = feature["messages"]
            image = feature["image"]
            
            formatted_messages = []
            for m in original_messages:
                if m["role"] == "user":
                    content = [
                        {"type": "image"}, 
                        {"type": "text", "text": m["content"]}
                    ]
                    formatted_messages.append({"role": "user", "content": content})
                else:
                    formatted_messages.append(m)
            
            text = self.processor.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(image)
            
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=8192, 
            return_tensors="pt"
        )
        
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        for i in range(len(input_ids)):
            start_indices = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
            if len(start_indices) > 0:
                last_turn_idx = start_indices[-1]
                labels[i, :last_turn_idx + 2] = -100
            
        batch["labels"] = labels
        return batch

# =============================================================================
# 4. Setup & Run
# =============================================================================

def setup_model_for_sft(model, processor):
    print("[Model Setup] Configuring Surgical Fine-Tuning strategy...")

    action_token_ids = [
        processor.tokenizer.convert_tokens_to_ids(t) 
        for t in ACTION_TOKENS
    ]
    action_token_ids = [idx for idx in action_token_ids if idx is not None]
    
    print(f"[Model Setup] Targeted Special Tokens: {len(action_token_ids)} tokens")
    
    model.enable_input_require_grads() 
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True

    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "visual" in name:
            param.requires_grad = False
        elif "embed_tokens" in name or "wte" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = True 
            trainable_params += param.numel()

    print(f"[Model Setup] Total Params: {total_params:,}")
    print(f"[Model Setup] Trainable Params (Approx): {trainable_params:,}")
    print("[Model Setup] Strategy: Vision(Frozen) + Embeds(Train) + LLM(Train)")
    
def run_sft():
    print(f"[SFT] Loading from {HP.INIT_MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.INIT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(HP.INIT_MODEL_PATH, trust_remote_code=True)
    
    setup_model_for_sft(model, processor)
    
    train_dataset = SFTDataset(HP.SFT_DATA_PATH)
    collator = SFTDataCollator(processor=processor)
    
    args = TrainingArguments(
        output_dir=HP.SFT_OUTPUT_PATH,
        num_train_epochs=HP.SFT_EPOCHS,
        per_device_train_batch_size=HP.SFT_BATCH_SIZE,
        gradient_accumulation_steps=HP.SFT_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_LEARN_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1
    )
    
    trainer = WeightedActionTrainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset, 
        data_collator=collator,
        processing_class=processor
    )
    
    print("[SFT] Starting Training with Action-Weighted Loss...")
    trainer.train()
    
    trainer.save_model(HP.SFT_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_OUTPUT_PATH)

if __name__ == "__main__":
    run_sft()