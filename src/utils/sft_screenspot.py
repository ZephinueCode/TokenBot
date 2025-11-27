# src/utils/sft_screenspot.py

import math
import os
import torch
from datasets import load_dataset
from typing import List, Tuple
from .parameters import HYPERPARAMS as HP
from .action_logic import MOVE_DELTAS

class ScreenSpotDataManager:
    """
    Manages loading, shuffling, and strict splitting of ScreenSpot data.
    Ensures no data leakage between Train, Eval, and Test.
    """
    def __init__(self):
        print(f"[ScreenSpot] Loading dataset from {HP.SCREENSPOT_DATA_PATH}...")
        try:
            # ScreenSpot usually allows 'test' split download
            ds = load_dataset(HP.SCREENSPOT_DATA_PATH, split="test") 
        except Exception as e:
            print(f"[Error] Failed to load ScreenSpot: {e}")
            self.raw_train, self.raw_eval, self.raw_test = [], [], []
            return

        # 1. Shuffle (Fixed Seed for reproducibility)
        ds = ds.shuffle(seed=HP.SFT_SEED)
        
        total_available = len(ds)
        limit = min(HP.SCREENSPOT_TOTAL_SIZE, total_available)
        ds = ds.select(range(limit))
        
        print(f"[ScreenSpot] Loaded {limit} samples.")

        # 2. Strict Split
        train_count = int(limit * HP.SCREENSPOT_TRAIN_RATIO)
        eval_count = int(limit * HP.SCREENSPOT_EVAL_RATIO)
        
        self.raw_train = ds.select(range(0, train_count))
        self.raw_eval = ds.select(range(train_count, train_count + eval_count))
        self.raw_test = ds.select(range(train_count + eval_count, limit))
        
        print(f"[ScreenSpot] Splits: Train={len(self.raw_train)}, Eval={len(self.raw_eval)}, Test={len(self.raw_test)}")

    def save_test_set(self, path="./data/screenspot_test.jsonl"):
        """Saves the raw test split for final evaluation (no training used)."""
        import json
        print(f"[ScreenSpot] Saving {len(self.raw_test)} test samples to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            for item in self.raw_test:
                # Save necessary metadata for evaluation
                bbox = item.get('bbox', [0,0,0,0])
                instruction = item['instruction']
                # We assume image loading happens via index later or raw dataset access
                entry = {"instruction": instruction, "bbox": bbox}
                f.write(json.dumps(entry) + "\n")

def get_shortest_path_actions(start_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Calculates greedy shortest path from start to target using discrete actions.
    Returns: List of (ActionToken, NewPosition)
    """
    cx, cy = start_pos
    tx, ty = target_pos
    path = []
    
    # Prioritize large jumps to be efficient
    valid_moves = [(k, v) for k, v in MOVE_DELTAS.items() if "MOVE" in k]
    valid_moves.sort(key=lambda x: math.hypot(x[1][0], x[1][1]), reverse=True)

    max_steps = 10 
    steps = 0
    
    while steps < max_steps:
        dist = math.hypot(tx - cx, ty - cy)
        
        # Click threshold (approx 20px radius)
        if dist <= 20: 
            break
            
        best_move_token = None
        best_new_pos = (cx, cy)
        min_dist_remaining = dist
        found_move = False
        
        for token, (mv_x, mv_y) in valid_moves:
            nx, ny = cx + mv_x, cy + mv_y
            # Boundary Check
            if not (0 <= nx < HP.IMAGE_SIZE and 0 <= ny < HP.IMAGE_SIZE):
                continue
            
            rem_dist = math.hypot(tx - nx, ty - ny)
            if rem_dist < min_dist_remaining:
                min_dist_remaining = rem_dist
                best_move_token = token
                best_new_pos = (nx, ny)
                found_move = True
        
        if found_move:
            path.append((best_move_token, best_new_pos))
            cx, cy = best_new_pos
            steps += 1
        else:
            break # Stuck or close enough
            
    # Final action is always Click
    path.append(("<CLICK_SHORT>", (cx, cy)))
    return path