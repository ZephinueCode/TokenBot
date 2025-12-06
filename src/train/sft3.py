# src/train/sft3.py

import torch
import random
import math
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from PIL import Image

from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)

# Project imports
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import ACTION_TOKENS
from ..utils.action_logic import MOVE_DELTAS
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..tools.visual_utils import draw_cursor
from ..utils.sft_screenspot_pro import ScreenSpotDataManager

# Reuse logic from Phase 2 (generate_cot_for_step NOW EXPECTS NO HISTORY ARG)
from .sft2 import generate_cot_for_step, WeightedActionTrainer

# =============================================================================
# 1. Local Helper: Dynamic Path Finder
# =============================================================================

def get_shortest_path_actions_dynamic(start_pos: Tuple[int, int], target_pos: Tuple[int, int], img_size: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Calculates greedy shortest path respecting DYNAMIC image boundaries.
    """
    cx, cy = start_pos
    tx, ty = target_pos
    w, h = img_size
    path = []
    
    # Prioritize large jumps to be efficient
    valid_moves = [(k, v) for k, v in MOVE_DELTAS.items() if "MOVE" in k]
    valid_moves.sort(key=lambda x: math.hypot(x[1][0], x[1][1]), reverse=True)

    max_steps = 10 
    steps = 0
    
    while steps < max_steps:
        dist = math.hypot(tx - cx, ty - cy)
        
        if dist <= 20: 
            break
            
        best_token = None
        best_pos = (cx, cy)
        min_dist = dist
        found = False
        
        for token, (mx, my) in valid_moves:
            nx, ny = cx + mx, cy + my
            
            # Boundary Check using ACTUAL image dimensions
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            
            rem = math.hypot(tx - nx, ty - ny)
            if rem < min_dist:
                min_dist = rem
                best_token = token
                best_pos = (nx, ny)
                found = True
        
        if found:
            path.append((best_token, best_pos))
            cx, cy = best_pos
            steps += 1
        else:
            break
            
    # Final action is always Click
    path.append(("<CLICK_SHORT>", (cx, cy)))
    return path

# =============================================================================
# 2. Dataset & Collator
# =============================================================================

@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images = [], []
        for f in features:
            msgs = f["messages"]
            fmt = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": msgs[1]["content"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": msgs[2]["content"]}]}
            ]
            txt = self.processor.apply_chat_template(fmt, tokenize=False, add_generation_prompt=False)
            texts.append(f"{AGENT_SYSTEM_PROMPT}\n{txt}")
            images.append(f["image"])
            
        batch = self.processor(
            text=texts, images=images, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i in range(len(labels)):
            starts = (batch["input_ids"][i] == im_start).nonzero(as_tuple=True)[0]
            if len(starts) >= 2:
                labels[i, :starts[-1] + 1] = -100 
        
        batch["labels"] = labels
        return batch

class ScreenSpotProSFTDataset(Dataset):
    def __init__(self, split="train"):
        self.ss_manager = ScreenSpotDataManager()
        self.split = split
        self.data_source = self.ss_manager.raw_train if split == "train" else self.ss_manager.raw_eval
        print(f"[SFT-3 Dataset] Split '{split}': {len(self.data_source)} base samples.")

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return self._process_sample(self.data_source[idx])

    def _process_sample(self, sample):
        # 1. Load Image from Path
        try:
            image_path = sample['image_path']
            # [NO RESIZE] Keep original resolution for maximum detail
            raw_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raw_img = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
        
        orig_w, orig_h = raw_img.size
        MAX_SIDE = 1280
        
        if max(orig_w, orig_h) > MAX_SIDE:
            scale = MAX_SIDE / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            raw_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            scale = 1.0 # No resizing needed
        
        w, h = raw_img.size
        
        # 2. Scale Target BBox
        bbox = sample.get('bbox', None)
        if bbox:
            # Coordinates are already absolute in ScreenSpot Pro
            abs_x1, abs_y1, abs_x2, abs_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            tx = int((abs_x1 + abs_x2) / 2)
            ty = int((abs_y1 + abs_y2) / 2)
            tx = max(0, min(w - 1, tx))
            ty = max(0, min(h - 1, ty))
        else:
            tx, ty = w // 2, h // 2

        # 3. Generate Trajectory Path (Dynamic)
        center_x, center_y = w // 2, h // 2
        full_path = get_shortest_path_actions_dynamic((center_x, center_y), (tx, ty), (w, h))
        
        # 4. Sampling Strategy with History
        if not full_path:
            curr_pos = (center_x, center_y)
            action_token = "<CLICK_SHORT>"
            history_tokens = []
        else:
            if random.random() < 0.33:
                step_idx = len(full_path) - 1 # Force final step
            else:
                step_idx = random.randint(0, max(0, len(full_path) - 2))
            
            target_step = full_path[step_idx]
            action_token = target_step[0]
            
            # --- Build History & Update Current State ---
            history_tokens = []
            curr_cx, curr_cy = center_x, center_y
            
            for i in range(step_idx):
                h_token, (nx, ny) = full_path[i]
                history_tokens.append(h_token)
                curr_cx, curr_cy = nx, ny
            
            curr_pos = (curr_cx, curr_cy)

        # 5. [FIXED] Format History for USER PROMPT
        if not history_tokens:
            history_str = "None (Start)"
        else:
            history_str = " -> ".join(history_tokens)
        
        # Construct User Input: Instruction + History
        user_content = f"[Action] Task: {sample['instruction']}\nPrevious Actions: {history_str}"

        # 6. Visualization & CoT
        image = draw_cursor(raw_img, curr_pos[0], curr_pos[1])
        
        # NOTE: history_tokens removed from generate_cot call (matching sft2 update)
        cot_response = generate_cot_for_step(
            cursor_pos=curr_pos, 
            target_pos=(tx, ty), 
            instruction=sample['instruction'], 
            next_action_token=action_token,
            img_size=(w, h)
        )
        
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            # [FIXED] User content now includes history
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": cot_response}
        ]
        
        return {"messages": msgs, "image": image}

# =============================================================================
# 3. Run Logic
# =============================================================================

def run_sft_screenspot_pro():
    input_path = HP.SFT_2_OUTPUT_PATH 
    output_path = getattr(HP, "SFT_3_OUTPUT_PATH", "./checkpoints/sft_phase3")
    
    if not os.path.exists(input_path):
        print(f"[Error] Stage 2 model not found at {input_path}")
        return

    print(f"[SFT-3] Loading Stage 2 Model from {input_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        input_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(input_path, trust_remote_code=True)
    
    # Freeze Vision
    model.enable_input_require_grads()
    model.get_input_embeddings().weight.requires_grad = True
    for name, param in model.named_parameters():
        if "visual" in name: param.requires_grad = False
        else: param.requires_grad = True
        
    train_ds = ScreenSpotProSFTDataset("train")
    eval_ds = ScreenSpotProSFTDataset("eval")
    collator = SFTDataCollator(processor)
    
    epochs = getattr(HP, "SFT_3_EPOCHS", 3)
    batch_size = getattr(HP, "SFT_3_BATCH_SIZE", 4)
    grad_accum = getattr(HP, "SFT_3_GRAD_ACCUM_STEPS", 8)
    lr = getattr(HP, "SFT_3_LEARN_RATE", 1e-5)

    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=240,
        eval_steps=240,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = WeightedActionTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, 
        data_collator=collator, processing_class=processor
    )
    
    print("[SFT-3] Starting ScreenSpot Pro Training (History in Input)...")
    trainer.train()
    
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    train_ds.ss_manager.save_test_set()

if __name__ == "__main__":
    run_sft_screenspot_pro()