# src/eval/evaluate_vlm.py

import torch
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from ..utils.parameters import HYPERPARAMS as HP
# Use the Pro version manager which handles local paths
from ..utils.sft_screenspot_pro import ScreenSpotDataManager
from ..tools.runner import Runner
from ..train.reward import batch_compute_rewards
from ..tools.visual_utils import visualize_trajectory

# --- Dynamic Imports for Baselines ---
try:
    from .baseline_vlm import BaselineAPIRunner
except ImportError:
    BaselineAPIRunner = None

try:
    from .baseline_grounding import BaselineGroundingRunner
except ImportError:
    BaselineGroundingRunner = None

def evaluate_model(mode="trained", limit=None, model_path=None, bbox_expansion=None):
    """
    Main Evaluation Loop.
    Args:
        mode: 
            - "trained": Local SFT Model
            - "api_baseline": General VLM Agent (GPT-4o, etc.)
            - "grounding_baseline": Qwen-VL Grounding Agent
        limit: Max samples to evaluate
        bbox_expansion: Float ratio (0.0 - 1.0) to expand GT bbox.
    """
    if model_path is None:
        model_path = HP.SFT_2_OUTPUT_PATH
    
    # [CONFIG] Set expansion ratio (default to 5%)
    if bbox_expansion is None:
        bbox_expansion = getattr(HP, "EVAL_BBOX_EXPANSION", 0.05)

    # 1. Setup Output Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = "./results/eval"
    eval_name = f"{mode}_{timestamp}_relax{int(bbox_expansion*100)}"
    result_dir = os.path.join(base_save_dir, eval_name)
    img_save_dir = os.path.join(result_dir, "images")
    
    os.makedirs(img_save_dir, exist_ok=True)
    print(f"\n[EVAL] Starting Evaluation: {mode.upper()}")
    print(f"[EVAL] BBox Relaxation: {bbox_expansion:.1%} of screen size")
    print(f"[EVAL] Saving results to: {result_dir}")

    # 2. Load Data (Use Test Split from Pro Manager)
    print(f"[EVAL] Loading ScreenSpot Test Set...")
    ss_manager = ScreenSpotDataManager()
    dataset = ss_manager.raw_test 
    
    if limit:
        print(f"[EVAL] Limiting to first {limit} samples.")
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"[EVAL] Samples to process: {len(dataset)}")

    # 3. Initialize Model & Runner
    model = None
    processor = None
    runner = None

    if mode == "api_baseline":
        if BaselineAPIRunner is None:
            raise ImportError("BaselineAPIRunner not found.")
        print(f"[EVAL] Initializing API Baseline Runner...")
        runner = BaselineAPIRunner()

    elif mode == "grounding_baseline":
        if BaselineGroundingRunner is None:
            raise ImportError("BaselineGroundingRunner not found.")
        print(f"[EVAL] Initializing Qwen-VL Grounding Runner...")
        runner = BaselineGroundingRunner()

    elif mode == "trained":
        print(f"[EVAL] Loading Local Model from: {model_path}")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            runner = Runner()
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            return
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 4. Main Loop
    results = []
    metrics = {
        "success_count": 0,
        "total_steps": 0,
        "total_reward": 0,
        "failed_reasons": {1:0, 2:0, 3:0, 4:0}
    }

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            # --- [ROBUST IMAGE LOADING] ---
            # Priority: "image_path" (Local) -> "image" (HF Object)
            raw_img = None
            
            if "image_path" in sample and sample["image_path"]:
                # Load from local path
                try:
                    raw_img = Image.open(sample["image_path"]).convert("RGB")
                except Exception as e:
                    print(f"[Warn] Failed to load path {sample['image_path']}: {e}")
            
            if raw_img is None and "image" in sample:
                # Fallback to HF dataset object
                img_data = sample["image"]
                if isinstance(img_data, Image.Image):
                    raw_img = img_data.convert("RGB")
                else:
                    # Try treating as path string if not PIL
                    try:
                        raw_img = Image.open(img_data).convert("RGB")
                    except: pass

            if raw_img is None:
                print(f"[Skip] No valid image found for sample {i}")
                continue

            w, h = raw_img.size
            
            # --- Prepare BBox ---
            bbox = sample.get('bbox', None)
            if not bbox and 'point' in sample:
                p = sample['point']
                bbox = [p[0], p[1], p[0], p[1]]
            
            # Scale BBox to Absolute Pixels
            final_bbox = [0, 0, 0, 0]
            if bbox:
                if all(0.0 <= x <= 1.0 for x in bbox):
                    # Normalized -> Absolute
                    final_bbox = [
                        bbox[0] * w, bbox[1] * h,
                        bbox[2] * w, bbox[3] * h
                    ]
                else:
                    # Already Absolute
                    final_bbox = bbox

            # Relax BBox Standards
            if bbox_expansion > 0 and final_bbox != [0,0,0,0]:
                margin_x = w * bbox_expansion
                margin_y = h * bbox_expansion
                
                final_bbox = [
                    max(0, final_bbox[0] - margin_x), # x1
                    max(0, final_bbox[1] - margin_y), # y1
                    min(w, final_bbox[2] + margin_x), # x2
                    min(h, final_bbox[3] + margin_y)  # y2
                ]

            # Construct GT Data Entry
            gt_data_entry = {
                "image": raw_img, 
                "instruction": sample['instruction'],
                "action_type": "click", 
                "bbox": final_bbox 
            }
            ground_truth_data = [gt_data_entry]

            # --- Run Inference ---
            if mode in ["api_baseline", "grounding_baseline"]:
                traj, _, _ = runner.run_trajectory(
                    input_text=sample['instruction'],
                    ground_truth_data=ground_truth_data,
                    max_steps=HP.EVAL_MAX_STEPS,
                )
            else:
                # Local Model
                traj, _, _ = runner.run_trajectory(
                    model=model,
                    processor=processor,
                    input_text=sample['instruction'],
                    ground_truth_data=ground_truth_data,
                    max_steps=HP.EVAL_MAX_STEPS, 
                    temperature=0.7 
                )

            # --- Metrics ---
            is_success = (traj.failed == 0)
            reward = batch_compute_rewards([traj])[0] 
            
            metrics["total_steps"] += traj.step_count
            metrics["total_reward"] += reward
            if is_success: 
                metrics["success_count"] += 1
            else:
                if traj.failed in metrics["failed_reasons"]:
                    metrics["failed_reasons"][traj.failed] += 1

            # --- Visualization ---
            vis_img = visualize_trajectory(
                base_image=raw_img,
                cursor_path=traj.cursor_path,
                actions=traj.tools,
                gt_bbox=final_bbox, 
                success=is_success,
                instruction=traj.global_question
            )
            
            status_str = "PASS" if is_success else "FAIL"
            img_filename = f"{mode}_id_{i:04d}_{status_str}_rew{reward:.1f}.png"
            vis_img.save(os.path.join(img_save_dir, img_filename))

            # --- Log Result ---
            results.append({
                "id": i,
                "instruction": sample['instruction'],
                "success": is_success,
                "steps": traj.step_count,
                "reward": reward,
                "fail_code": traj.failed,
                "history": traj.tools,
                "vis_file": img_filename
            })
            
            if i % 10 == 0: torch.cuda.empty_cache()

        except Exception as e:
            print(f"[EVAL] Error processing sample {i}: {e}")
            continue

    # 5. Final Report
    count = len(dataset)
    if count == 0: count = 1
    
    acc = metrics["success_count"] / count
    avg_steps = metrics["total_steps"] / count
    avg_rew = metrics["total_reward"] / count
    
    print("\n" + "="*40)
    print(f"EVAL REPORT: {mode.upper()}")
    print(f"Relaxation:    {bbox_expansion:.1%}")
    print(f"Total Samples: {count}")
    print(f"Success Rate:  {acc:.2%}")
    print(f"Avg Steps:     {avg_steps:.2f}")
    print(f"Avg Reward:    {avg_rew:.2f}")
    print(f"Failure Stats: {metrics['failed_reasons']}")
    print("="*40 + "\n")

    # Save JSON
    report = {
        "meta": {
            "mode": mode,
            "model_path": model_path if mode == "trained" else mode,
            "timestamp": timestamp,
            "relaxation_ratio": bbox_expansion,
            "num_samples": count
        },
        "metrics": {
            "accuracy": acc,
            "avg_steps": avg_steps,
            "avg_reward": avg_rew,
            "failed_breakdown": metrics["failed_reasons"]
        },
        "details": results
    }
    
    json_path = os.path.join(result_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"[EVAL] Full report saved to: {json_path}")

if __name__ == "__main__":
    # Example Usage:
    
    # 1. Evaluate Grounding Baseline (Qwen-VL)
    # evaluate_model(mode="grounding_baseline", limit=20, bbox_expansion=0.05)
    
    # 2. Evaluate API Baseline (GPT-4o)
    # evaluate_model(mode="api_baseline", limit=10, bbox_expansion=0.05)
    
    # 3. Evaluate Local SFT Model (Phase 3)
    evaluate_model(mode="trained", limit=20, bbox_expansion=0.05, model_path=HP.SFT_2_OUTPUT_PATH)