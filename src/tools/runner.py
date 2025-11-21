# src/tools/runner.py

import torch
import numpy as np
import os 
from PIL import Image
from typing import List, Dict, Tuple, Any
from io import BytesIO

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action_logic import MOVE_DELTAS, get_action_type
from .visual_utils import draw_cursor, visualize_trajectory 
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.action import ACTION_TOKENS

# =============================================================================
# DATA STRUCTURES (Unchanged)
# =============================================================================

class ToolData:
    """Simple container for passing data around."""
    def __init__(self, image: Image.Image, text: str):
        self.image = image
        self.text = text
    
    def getText(self): return self.text
    def getImage(self): return self.image

class GTStep:
    """Represents ONE step in the Ground Truth path."""
    def __init__(self, image: Image.Image, bbox: List[float], instruction: str, action_type: str):
        self.image = image
        self.bbox = bbox
        self.instruction = instruction
        self.action_type = action_type

class GTTrajectory:
    """Holds the Ideal Ground Truth path."""
    def __init__(self, gt_data: List[Dict]):
        self.steps = []
        
        for item in gt_data:
            img_data = item['image']
            
            if isinstance(img_data, dict):
                if 'bytes' in img_data and img_data['bytes'] is not None:
                    image = Image.open(BytesIO(img_data['bytes'])).convert("RGB")
                else:
                    image = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
            elif isinstance(img_data, Image.Image):
                image = img_data
            elif isinstance(img_data, np.ndarray):
                image = Image.fromarray(img_data).convert("RGB")
            else:
                image = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))

            self.steps.append(GTStep(
                image=image,
                bbox=item['bbox'],
                instruction=item['instruction'],
                action_type=item['action_type']
            ))
            
        self.total_steps = len(self.steps)

    def get_step(self, idx: int) -> GTStep:
        if idx < self.total_steps:
            return self.steps[idx]
        return None

class AgentTrajectory:
    """
    Holds the dynamic execution path of the Agent.
    Maintains state, history, path coordinates, and pass/fail status.
    """
    def __init__(self, input_text: str, start_gt_step: GTStep, total_gt_steps: int = 1):
        self.global_question = input_text
        
        # --- Task Info ---
        self.total_gt_steps = total_gt_steps 
        
        # --- State ---
        self.current_gt_step_idx = 0
        self.current_base_image = start_gt_step.image.copy() 
        
        # [CRITICAL] Store the currently active GT target BBox for Reward calculation
        self.target_bbox = start_gt_step.bbox 
        
        # Init Cursor at Center of the CURRENT image
        w, h = self.current_base_image.size
        self.cursor_pos = [w // 2, h // 2]
        
        # --- Evaluation Status ---
        self.failed = 0 
        self.gt_steps_passed = 0
        
        # --- History ---
        self.step_count = 0
        self.tools = []       
        self.images = []      
        
        # Visual Path Recording for Reward shaping
        self.cursor_path = [tuple(self.cursor_pos)]
        
        # Render the initial frame (Step 0 state)
        self._update_visual_history()

    def _update_visual_history(self):
        """Draws cursor on current base image and appends to history."""
        clean_copy = self.current_base_image.copy()
        viz_img = draw_cursor(
            clean_copy, 
            int(self.cursor_pos[0]), 
            int(self.cursor_pos[1])
        )
        self.images.append(viz_img)

    def get_current_view(self) -> Image.Image:
        return self.images[-1]

    def move_cursor(self, dx, dy):
        w, h = self.current_base_image.size
        cx, cy = self.cursor_pos
        
        nx = max(0, min(cx + dx, w - 1))
        ny = max(0, min(cy + dy, h - 1))
        
        self.cursor_pos = [nx, ny]
        self.cursor_path.append((nx, ny))
        
        self._update_visual_history()

    def advance_to_next_gt_step(self, next_gt_step: GTStep):
        """Moves Agent to the next GT screen state."""
        self.current_gt_step_idx += 1
        self.gt_steps_passed += 1
        
        self.current_base_image = next_gt_step.image.copy()
        self.target_bbox = next_gt_step.bbox
        
        # Reset cursor to center of the NEW image
        w, h = self.current_base_image.size
        self.cursor_pos = [w // 2, h // 2]
        
        # Record Reset point as a jump in path
        self.cursor_path.append((w // 2, h // 2))
        self._update_visual_history()

# =============================================================================
# RUNNER (Main Execution Logic)
# =============================================================================

class Runner:
    def __init__(self):
        print(f"[Runner] Initialized. CoT & Dynamic Constraint Enabled.")

    def check_hit(self, cursor_pos, bbox, img_size):
        """
        Checks if the cursor position hits the target bounding box.
        [FIX]: Automatically scales normalized BBox (0.0-1.0) to absolute pixels.
        """
        cx, cy = cursor_pos
        w, h = img_size
        x1, y1, x2, y2 = bbox
        
        # [Fix] Detect if bbox is normalized (all coordinates <= 1.0)
        # If so, scale up by image dimensions
        if all(0.0 <= c <= 1.0 for c in [x1, y1, x2, y2]):
            x1 *= w
            x2 *= w
            y1 *= h
            y2 *= h

        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def run_trajectory(
        self,
        model,
        processor,
        input_text: str,
        ground_truth_data: List[Dict],
        max_steps: int = 10,
        temperature: float = 0.0 
    ) -> Tuple[AgentTrajectory, List[int], List[float]]:
        
        # 1. Initialize State
        gt_traj = GTTrajectory(ground_truth_data)
        
        # Pass total_gt_steps to AgentTrajectory
        agent_traj = AgentTrajectory(
            input_text, 
            gt_traj.get_step(0), 
            total_gt_steps=gt_traj.total_steps
        )
        
        # 2. Define Constraints
        valid_action_ids = {processor.tokenizer.convert_tokens_to_ids(t) for t in ACTION_TOKENS}
        angle_id = processor.tokenizer.convert_tokens_to_ids("<")
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        allowed_ids_set = valid_action_ids.union({angle_id, im_end_id})
        
        all_token_ids = [] 
        all_logprobs = []

        # === ACTION LOOP ===
        try: # Global try-block for CUDA OOM protection
            for step in range(max_steps):
                agent_traj.step_count += 1
                current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
                curr_img = agent_traj.get_current_view()

                # Build Prompt
                messages = [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": curr_img},
                            {"type": "text", "text": f"[Action] {agent_traj.global_question}\n"}
                        ]
                    }
                ]

                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text_prompt],
                    images=[curr_img],
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)
                
                # Generate
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                pixel_values = inputs.pixel_values
                image_grid_thw = inputs.image_grid_thw
                
                generated_text = ""
                final_action_token = None
                final_action_logprob = 0.0
                final_action_id = None 
                
                MAX_NEW_TOKENS = 512
                
                with torch.no_grad():
                    for gen_i in range(MAX_NEW_TOKENS):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw
                        )
                        logits = outputs.logits[0, -1, :]
                        
                        # Constraint
                        if "Action: " in generated_text:
                            mask = torch.full_like(logits, -1e9)
                            for allow_id in allowed_ids_set:
                                if allow_id < len(logits):
                                    mask[allow_id] = 0.0
                            logits = logits + mask
                        
                        # Sampling
                        if temperature == 0.0:
                            next_token_id = torch.argmax(logits).item()
                        else:
                            scaled_logits = logits / temperature
                            probs = torch.softmax(scaled_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, 1).item()
                        
                        step_logprob = torch.log_softmax(logits, dim=-1)[next_token_id].item()

                        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
                        attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=model.device)], dim=1)
                        
                        next_token_text = processor.decode([next_token_id], skip_special_tokens=False)
                        generated_text += next_token_text
                        
                        if next_token_text in ACTION_TOKENS:
                            final_action_token = next_token_text
                            final_action_logprob = step_logprob
                            final_action_id = next_token_id
                            break 
                        
                        if next_token_text in ["<|im_end|>", "<|endoftext|>"]:
                            break

                # Parse
                parts = generated_text.split("Action:")
                reasoning_str = parts[0].replace("Reasoning:", "").strip()
                
                print(f"\n[Step {step+1}]")
                print(f"  Thinking: {reasoning_str[:100]}...")
                
                if final_action_token:
                    print(f"  Action:   {final_action_token}")
                    clean_gen_text = generated_text.strip()
                    agent_traj.tools.append(clean_gen_text)
                    all_token_ids.append(final_action_id) 
                    all_logprobs.append(final_action_logprob)
                    token_text = final_action_token
                else:
                    print(f"  Action:   [FAILED] {generated_text[-20:]}")
                    agent_traj.failed = 1 
                    agent_traj.tools.append("INVALID")
                    all_token_ids.append(0)
                    all_logprobs.append(-100.0)
                    break
                
                # Execute
                token_action_type = get_action_type(token_text)
                
                if token_action_type == "move":
                    if token_text in MOVE_DELTAS:
                        dx, dy = MOVE_DELTAS[token_text]
                        agent_traj.move_cursor(dx, dy)
                    else:
                        agent_traj.failed = 1; break
                
                elif token_action_type in ["click", "scroll", "text", "nav"]:
                    if token_action_type != current_gt.action_type:
                        agent_traj.failed = 2; break
                    
                    # [FIX]: Pass image size for auto-scaling normalized bbox
                    if self.check_hit(agent_traj.cursor_pos, current_gt.bbox, curr_img.size):
                        print("    -> Hit! Target reached.")
                        if agent_traj.current_gt_step_idx < gt_traj.total_steps - 1:
                            next_step = gt_traj.get_step(agent_traj.current_gt_step_idx + 1)
                            agent_traj.advance_to_next_gt_step(next_step)
                        else:
                            agent_traj.current_gt_step_idx += 1
                            agent_traj.gt_steps_passed += 1
                            print("    -> Task Success (Action Hit).")
                            agent_traj.failed = 0 
                            break 
                    else:
                        print("    -> Miss! Position off.")
                        agent_traj.failed = 3; break
                
                elif token_action_type == "end":
                    if agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                        print("    -> Task Success.")
                        agent_traj.failed = 0
                    else:
                        print("    -> Premature Stop.")
                        agent_traj.failed = 4
                    break
                else:
                    agent_traj.failed = 1; break
        
        except BaseException:
            print("\n[Runner] Exception! Aborting trajectory safely.")
            torch.cuda.empty_cache()
            agent_traj.failed = 4 
            if len(all_token_ids) < len(agent_traj.tools):
                 pass

        # Timeout Logic
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             if agent_traj.current_gt_step_idx < gt_traj.total_steps:
                 agent_traj.failed = 4 

        # =====================================================================
        # REAL-TIME MONITORING WITH INSTRUCTION OVERLAY
        # =====================================================================
        try:
            os.makedirs("./results", exist_ok=True)
            base_clean = gt_traj.get_step(0).image
            
            # Handle BBox for visualization (also check scaling)
            viz_bbox = agent_traj.target_bbox
            w, h = base_clean.size
            if viz_bbox and all(0.0 <= c <= 1.0 for c in viz_bbox):
                viz_bbox = [viz_bbox[0]*w, viz_bbox[1]*h, viz_bbox[2]*w, viz_bbox[3]*h]

            is_success = (agent_traj.failed == 0)
            
            vis_img = visualize_trajectory(
                base_image=base_clean,
                cursor_path=agent_traj.cursor_path,
                actions=agent_traj.tools,
                gt_bbox=viz_bbox, # Use scaled bbox
                success=is_success,
                instruction=agent_traj.global_question
            )
            vis_img.save("./results/current.png")
        except Exception as e:
            print(f"[Runner] Monitor visualization failed: {e}")
        
        torch.cuda.empty_cache()

        return agent_traj, all_token_ids, all_logprobs