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
# 0. Custom Trainer for Action Reweighting (FIXED)
# =============================================================================

class WeightedActionTrainer(Trainer):
    """
    Custom Trainer to solve 'Loss Dilution'.
    Standard CrossEntropyLoss averages all tokens. Since Reasoning is long (~100 tokens)
    and Action is short (1 token), the model ignores the Action accuracy.
    
    This Trainer forces a high weight (e.g., 20x) on the FINAL token (the Action).
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        # 1. Forward Pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 2. Shift Logits and Labels for Causal LM
        # logits[t] predicts labels[t+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 3. Compute Per-Token Loss (No Reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        batch_size = labels.size(0)
        
        # [FIX] Get vocab_size dynamically from logits shape
        # This avoids AttributeError if config doesn't expose vocab_size
        vocab_size = shift_logits.size(-1)
        
        # Flatten dimensions to compute standard CE
        raw_loss = loss_fct(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1)
        )
        # Reshape back to [Batch, Seq_Len-1]
        raw_loss = raw_loss.view(batch_size, -1)

        # 4. Build Weight Matrix
        weights = torch.ones_like(raw_loss)
        
        # Config: How much more important is the Action token?
        ACTION_WEIGHT = 20.0 
        
        # Iterate over batch to find the end of each sequence
        for i in range(batch_size):
            # Get indices of non-padding labels (-100 is padding/masked)
            valid_mask = (shift_labels[i] != -100)
            valid_indices = valid_mask.nonzero(as_tuple=False)
            
            if len(valid_indices) > 0:
                # The last valid index is our Target Action Token 
                # (Because the data generator puts "Action: <TOKEN>" at the very end)
                last_idx = valid_indices[-1].item()
                
                # Apply weight to exactly this 1 token (Window = 1)
                weights[i, last_idx] = ACTION_WEIGHT

        # 5. Compute Weighted Mean
        # Mask out padding tokens from the weights so they don't affect the mean
        valid_tokens_mask = (shift_labels != -100).float()
        final_weights = weights * valid_tokens_mask
        
        # Weighted Average = Sum(Loss * Weight) / Sum(Weights)
        loss = (raw_loss * final_weights).sum() / (final_weights.sum() + 1e-8)

        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 1. Semantic Grid & Logic
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
    Ends exactly with the Action Token.
    """
    # === Navigation Logic ===
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
    
    reasoning = "Reasoning: "
    reasoning += f"The cursor is currently in the **{c_region}** region. "
    reasoning += f"The target '{target_name}' is located in the **{t_region}** region. "

    # === [NEW] Relative Position Logic ===
    if c_region == t_region:
        reasoning += f"I need to examine the position more carefully. "
    
    dx = tx - cx
    dy = ty - cy
    margin = 15
    
    v_rel = ""
    h_rel = ""

    if dy < -margin:
        v_rel = "Top"
    elif dy > margin:
        v_rel = "Bottom"
    
    if dx < -margin:
        h_rel = "Left"
    elif dx > margin:
        h_rel = "Right"
    
    rel_pos_str = ""
    if v_rel and h_rel:
        rel_pos_str = f"{v_rel}-{h_rel}"  # Top-Left, Bottom-Right
    elif v_rel:
        rel_pos_str = v_rel               # Top
    elif h_rel:
        rel_pos_str = h_rel               # Right
    else:
        rel_pos_str = "Overlapping"

    if rel_pos_str != "Overlapping":
        reasoning += f"The target is to the **{rel_pos_str}** of the cursor. "
    else:
        reasoning += f"The cursor is right on the target. "
    
    HIT_THRESHOLD = 15
    action = ""
    
    if dist < HIT_THRESHOLD:
        reasoning += "The cursor is aligned with the target. "
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
        
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
            d_val = abs(dx)
            suffix = "FAR" if d_val > 250 else ("MID" if d_val > 50 else "CLO")
            
            if suffix == "FAR": reasoning += "There is a significant gap. I need a large jump. "
            elif suffix == "MID": reasoning += "The target is moderately away. I need a standard step. "
            else: reasoning += "The target is very close. I need a micro-adjustment. "

            action = f"<MOVE_{direction}_{suffix}>"
        else:
            direction = "DOWN" if dy > 0 else "UP"
            d_val = abs(dy)
            suffix = "FAR" if d_val > 250 else ("MID" if d_val > 50 else "CLO")
            
            if suffix == "FAR": reasoning += "There is a significant gap. I need a large jump. "
            elif suffix == "MID": reasoning += "The target is moderately away. I need a standard step. "
            else: reasoning += "The target is very close. I need a micro-adjustment. "
            
            action = f"<MOVE_{direction}_{suffix}>"
            
    return f"{reasoning}\nAction: {action}"

# =============================================================================
# 2. Dataset (MODIFIED: 5:3:2 Split + Prompt Diversity)
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
        
        self.ui_names = ["Submit", "Cancel", "Search", "Menu", "Settings", "Back", "Profile", "Login"]
        self.text_contents = ["hello", "test", "123456", "user", "admin"]
        
        # [NEW] Diverse Prompt Templates for better generalization
        self.prompts_click = [
            "Click {name}", "Select {name}", "Tap on {name}", "Hit the {name} button", "Choose {name}"
        ]
        self.prompts_move = [
            "Move cursor to {name}", "Hover over {name}", "Point at {name}", "Go to {name}", "Find {name}"
        ]
        self.prompts_text = [
            "Type '{content}' into {name}", "Enter '{content}' in {name}", "Fill {name} with '{content}'", "Input '{content}'"
        ]
        self.prompts_back = [
            "Go back", "Return to previous page", "Back", "Navigate back"
        ]
        self.prompts_home = [
            "Go home", "Return to main screen", "Home", "Main menu", "Exit to desktop"
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

        # === TYPE 2: SPATIAL & NAVIGATION (Synthetic) - 5:3:2 Split ===
        else:
            image = self.get_random_background()
            draw = ImageDraw.Draw(image)
            margin = 50
            
            sample_rng = random.random()
            
            tx = random.randint(margin, HP.IMAGE_SIZE - margin)
            ty = random.randint(margin, HP.IMAGE_SIZE - margin)
            target_name = random.choice(self.ui_names)
            
            # ----------------------------------------------------------
            # CASE A: Moving (50%) - Force <MOVE>
            # Cursor: FAR from target (>100px)
            # Intent: Random (User wants Click/Move/Text, but needs to move first)
            # ----------------------------------------------------------
            if sample_rng < 0.5:
                # Ensure cursor is far enough
                while True:
                    cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                    cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                    if math.hypot(cx-tx, cy-ty) > 100: break
                
                # Intent can be varied, instructions reflect user goal
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

            # ----------------------------------------------------------
            # CASE B: Direct Clicking (30%) - Force <CLICK>
            # Cursor: ON target (<15px), may require slight adjustments.
            # Intent: Click
            # ----------------------------------------------------------
            elif sample_rng < 0.8:
                cx = tx + random.randint(-20, 20)
                cy = ty + random.randint(-20, 20)
                intent = {"type": "click", "name": target_name}
                instr = random.choice(self.prompts_click).format(name=target_name)

            # ----------------------------------------------------------
            # CASE C: Other Tasks (20%) - Text & Nav
            # ----------------------------------------------------------
            else:
                # Split 50/50 between Text (Ready) and Nav
                if random.random() < 0.5:
                    # Text Ready (Cursor ON target)
                    cx = tx + random.randint(-10, 10)
                    cy = ty + random.randint(-10, 10)
                    content = random.choice(self.text_contents)
                    intent = {"type": "text", "name": target_name, "content": content}
                    instr = random.choice(self.prompts_text).format(name=target_name, content=content)
                else:
                    # Navigation (Cursor random, Target irrelevant/invisible)
                    intent_mode = random.choice(["back", "home"])
                    intent = {"type": "nav", "mode": intent_mode, "name": "System"}
                    cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                    cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                    tx, ty = 0, 0 # Hide target
                    
                    if intent_mode == "back":
                        instr = random.choice(self.prompts_back)
                    else:
                        instr = random.choice(self.prompts_home)

            # --- Drawing Logic ---
            # Only draw target for non-Nav tasks
            if intent.get("type") != "nav":
                draw.rectangle([tx-30, ty-20, tx+30, ty+20], outline="green", width=3)
                try: draw.text((tx-20, ty-35), target_name, fill="green")
                except: pass
            
            # Draw Cursor
            draw.line([(cx-15, cy), (cx+15, cy)], fill="red", width=4)
            draw.line([(cx, cy-15), (cx, cy+15)], fill="red", width=4)
            
            # Draw Distractors (Noise)
            for _ in range(random.randint(2, 5)):
                dx = random.randint(margin, HP.IMAGE_SIZE - margin)
                dy = random.randint(margin, HP.IMAGE_SIZE - margin)
                if abs(dx - tx) > 60 or abs(dy - ty) > 60:
                    draw.rectangle([dx-30, dy-20, dx+30, dy+20], outline="blue", width=2)

            # Generate Label
            cot_response = get_spatial_label_with_cot((cx, cy), (tx, ty), intent)
            
            msgs = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"[Action] {instr}"},
                {"role": "assistant", "content": cot_response}
            ]
            
            return {"messages": msgs, "image": image}

# =============================================================================
# 3. Collator (Unchanged)
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
            max_length=2048, 
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

    # 1. Identify Target Token IDs (Special Tokens)
    action_token_ids = [
        processor.tokenizer.convert_tokens_to_ids(t) 
        for t in ACTION_TOKENS
    ]
    action_token_ids = [idx for idx in action_token_ids if idx is not None]
    
    print(f"[Model Setup] Targeted Special Tokens: {len(action_token_ids)} tokens")
    
    target_indices_tensor = torch.tensor(action_token_ids)

    # 2. Enable Gradients for Embeddings
    model.enable_input_require_grads() 
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True

    # 4. Configure Freezing
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if "visual" in name:
            param.requires_grad = False # Freeze Vision
        elif "embed_tokens" in name or "wte" in name:
            # Physically unfrozen, but logically restricted by hook
            hidden_dim = param.shape[1]
            trainable_params += len(action_token_ids) * hidden_dim
        else:
            param.requires_grad = True # Train LLM for Reasoning
            trainable_params += param.numel()

    print(f"[Model Setup] Total Params: {total_params:,}")
    print(f"[Model Setup] Trainable Params (Approx): {trainable_params:,}")
    print("[Model Setup] Strategy: Vision(Freeze) + Embeds(Surgical) + LLM(Train)")
    
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
    
    # Use Custom Weighted Trainer instead of Standard Trainer
    trainer = WeightedActionTrainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset, 
        data_collator=collator
    )
    
    print("[SFT] Starting Training with Action-Weighted Loss...")
    trainer.train()
    
    trainer.save_model(HP.SFT_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_OUTPUT_PATH)

if __name__ == "__main__":
    run_sft()