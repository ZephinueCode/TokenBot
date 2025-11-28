# src/utils/sft_screenspot_pro.py

import math
import os
import json
import torch
from glob import glob
from PIL import Image
from datasets import Dataset as HFDataset
from typing import List, Tuple, Dict, Any
from .parameters import HYPERPARAMS as HP
from .action_logic import MOVE_DELTAS

class ScreenSpotDataManager:
    """
    Manages loading, processing, and splitting of the local ScreenSpot Pro dataset.
    Expects structure:
        root/
          ├── images/
          └── annotations/ (*.json)
    """
    def __init__(self):
        self.data_path = getattr(HP, "SCREENSPOT_PRO_PATH", "./data/screenspot_pro")
        # Define max dimension size
        self.max_image_dim = 1920
        # Create a cache directory for resized images to avoid modifying originals
        self.resized_dir = os.path.join(self.data_path, f"images_resized_{self.max_image_dim}")
        os.makedirs(self.resized_dir, exist_ok=True)

        print(f"[ScreenSpot Pro] Loading dataset from {self.data_path}...")
        
        self.samples = self._load_local_data()
        
        if not self.samples:
            print("[Error] No data found. Check path and structure.")
            self.raw_train, self.raw_eval, self.raw_test = [], [], []
            return

        # 1. Convert to HF Dataset for easy manipulation
        ds = HFDataset.from_list(self.samples)

        # 2. Shuffle (Fixed Seed)
        ds = ds.shuffle(seed=HP.SFT_SEED)
        
        total_available = len(ds)
        limit = min(getattr(HP, "SCREENSPOT_PRO_TOTAL_SIZE", total_available), total_available)
        ds = ds.select(range(limit))
        
        print(f"[ScreenSpot Pro] Loaded {limit} samples.")

        # 3. Strict Split
        # Default ratios: Train 0.8, Eval 0.1, Test 0.1 if not specified
        train_ratio = getattr(HP, "SCREENSPOT_TRAIN_RATIO", 0.8)
        eval_ratio = getattr(HP, "SCREENSPOT_EVAL_RATIO", 0.1)
        
        train_count = int(limit * train_ratio)
        eval_count = int(limit * eval_ratio)
        
        self.raw_train = ds.select(range(0, train_count))
        self.raw_eval = ds.select(range(train_count, train_count + eval_count))
        self.raw_test = ds.select(range(train_count + eval_count, limit))
        
        print(f"[ScreenSpot Pro] Splits: Train={len(self.raw_train)}, Eval={len(self.raw_eval)}, Test={len(self.raw_test)}")

    def _resize_image_and_bbox(self, original_path: str, bbox: List[float], original_rel_path: str) -> Tuple[str, List[float], List[int]]:
        """
        Resizes image if strictly larger than max_image_dim. 
        Scales bbox coordinates accordingly.
        Returns: (new_image_path, new_bbox, new_img_size)
        """
        try:
            # Check if we already processed this image to save time
            base_name = os.path.basename(original_rel_path)
            # Use a flattened path structure or keep hierarchy depending on preference. 
            # Here we just use the filename to keep it simple in the flat resize folder.
            save_name = f"{os.path.splitext(base_name)[0]}_resized.png"
            save_path = os.path.join(self.resized_dir, save_name)

            # Open original to check size
            with Image.open(original_path) as img:
                w, h = img.size
                
                # If image is already small enough, return original data
                if w <= self.max_image_dim and h <= self.max_image_dim:
                    return original_path, bbox, [w, h]

                # Calculate scaling factor
                scale = min(self.max_image_dim / w, self.max_image_dim / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                # If the processed file already exists on disk, we can skip saving,
                # BUT we must re-calculate bbox since we don't store processed bboxes separately here.
                # If you trust the cache, you just need the scale. 
                
                # Calculate new bbox
                # bbox format usually: [xmin, ymin, xmax, ymax]
                new_bbox = [coord * scale for coord in bbox]

                # Only save if not exists
                if not os.path.exists(save_path):
                    # Resize and Save
                    print(f"Resizing {base_name}: ({w}x{h}) -> ({new_w}x{new_h})")
                    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    img_resized.save(save_path)
                
                return save_path, new_bbox, [new_w, new_h]

        except Exception as e:
            print(f"[Warning] Resize failed for {original_path}: {e}")
            # Fallback to original if resize fails
            return original_path, bbox, [0, 0]

    def _load_local_data(self) -> List[Dict[str, Any]]:
        """Parses JSON annotations, verifies images, and handles Resizing."""
        img_dir = os.path.join(self.data_path, "images")
        ann_dir = os.path.join(self.data_path, "annotations")
        
        if not os.path.exists(ann_dir):
            print(f"[Error] Annotation dir not found: {ann_dir}")
            return []

        samples = []
        json_files = glob(os.path.join(ann_dir, "*.json"))
        
        print(f"[Data] Found {len(json_files)} annotation files. Processing images...")
        
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        items = data.values()
                    elif isinstance(data, list):
                        items = data
                    else:
                        continue

                    for item in items:
                        rel_path = item.get("img_filename", "")
                        abs_path = os.path.join(img_dir, rel_path)
                        
                        if os.path.exists(abs_path):
                            raw_bbox = item.get("bbox", [0, 0, 0, 0])
                            
                            # --- Resize Logic Added Here ---
                            final_path, final_bbox, final_size = self._resize_image_and_bbox(
                                abs_path, 
                                raw_bbox,
                                rel_path
                            )
                            # -------------------------------

                            entry = {
                                "image_path": final_path, 
                                "bbox": final_bbox, # Updated BBox
                                "instruction": item.get("instruction", ""),
                                "instruction_cn": item.get("instruction_cn", ""),
                                "id": item.get("id", "unknown"),
                                "img_size": final_size # Updated Size
                            }
                            samples.append(entry)
            except Exception as e:
                print(f"[Warning] Failed to parse {jf}: {e}")
        
        return samples

    def save_test_set(self, path="./data/screenspot_pro_test.jsonl"):
        """Saves the raw test split for final evaluation."""
        import json
        print(f"[ScreenSpot Pro] Saving {len(self.raw_test)} test samples to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in self.raw_test:
                entry = {
                    "image_path": item['image_path'],
                    "instruction": item['instruction'],
                    "bbox": item['bbox']
                }
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