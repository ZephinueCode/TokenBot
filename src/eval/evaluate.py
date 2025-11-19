import torch
import os
import json
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image # Need to handle image extraction if needed

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.dataset import prepare_grpo1_dataset
from ..tools.runner import Runner, ToolData, AgentTrajectory
from ..train.reward import batch_compute_rewards
from .baseline_vlm import BaselineAPIRunner
from ..tools.visual_utils import visualize_trajectory # <--- Import Viz

def evaluate_model(mode="trained", num_samples=None):
    # ... (Init code same as before) ...
    dataset = prepare_grpo1_dataset(HP.EVAL_DATA_PATH, size=num_samples)
    
    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(HP.EVAL_OUTPUT_DIR, f"{mode}_{timestamp}")
    img_save_dir = os.path.join(result_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)
    
    if mode == "api_baseline":
        print(f"[EVAL] Mode: API BASELINE")
        runner = BaselineAPIRunner()
        model = None
        processor = None
    else:
        model_path = HP.GRPO1_OUTPUT_PATH 
        print(f"[EVAL] Mode: TRAINED MODEL ({model_path})")
        runner = Runner()
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Error: {e}")
            return
            
    print(f"[EVAL] Dataset size: {len(dataset)}")
    
    results = []
    success_count = 0
    total_steps = 0
    total_reward = 0
    
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # Run Trajectory
        if mode == "api_baseline":
            traj, _, _ = runner.run_trajectory(
                input_text=sample['question'],
                ground_truth_data=sample['ground_truth_traj'],
                max_steps=HP.EVAL_MAX_STEPS
            )
        else:
            tool_input = ToolData(image=sample['image'], text=sample['question'])
            traj, _, _ = runner.run_trajectory(
                model=model,
                processor=processor,
                input_text=sample['question'], 
                ground_truth_data=sample['ground_truth_traj'],
                max_steps=HP.EVAL_MAX_STEPS,
                temperature=0.0
            )
        
        # Metrics
        is_success = (traj.failed == 0)
        if is_success: success_count += 1
        total_steps += traj.step_count
        reward = batch_compute_rewards([traj])[0]
        total_reward += reward
        
        # === VISUALIZATION LOGIC ===
        # Get base image (Step 0)
        # Note: sample['image'] is already a PIL image from dataset loader
        base_img = sample['image'].convert("RGB")
        
        # Get GT BBox (Step 0)
        gt_bbox = sample['ground_truth_traj'][0]['bbox']
        
        # Draw
        viz_img = visualize_trajectory(
            base_image=base_img,
            cursor_path=traj.cursor_path, 
            actions=traj.tools,
            gt_bbox=gt_bbox,
            success=is_success
        )
        
        # Save
        status_str = "PASS" if is_success else "FAIL"
        img_filename = f"{i:04d}_{status_str}.png"
        viz_img.save(os.path.join(img_save_dir, img_filename))
        # ===========================

        results.append({
            "id": i,
            "instruction": sample['question'],
            "success": is_success,
            "steps": traj.step_count,
            "reward": reward,
            "fail_reason": traj.failed,
            "history": traj.tools,
            "vis_image": img_filename
        })
        
    # Summary
    count = len(dataset)
    accuracy = success_count / count
    avg_steps = total_steps / count
    avg_reward = total_reward / count
    
    print("\n" + "="*40)
    print(f"EVAL REPORT: {mode.upper()}")
    print(f"Accuracy:   {accuracy:.2%}")
    print(f"Avg Steps:  {avg_steps:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print("="*40 + "\n")
    
    # Save JSON
    json_path = os.path.join(result_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump({
            "meta": {"mode": mode, "samples": count},
            "metrics": {"accuracy": accuracy, "avg_steps": avg_steps, "avg_reward": avg_reward},
            "details": results
        }, f, indent=2)
    
    print(f"Results saved to {result_dir}")

if __name__ == "__main__":
    evaluate_model(mode="trained", num_samples=HP.EVAL_DATASET_SIZE)