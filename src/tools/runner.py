import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from io import BytesIO

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.action_logic import MOVE_DELTAS, get_action_type
from .visual_utils import draw_cursor

# =============================================================================
# DATA STRUCTURES
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
        self.action_type = action_type # e.g., "click", "scroll", "type"

class GTTrajectory:
    """
    Holds the Ideal Ground Truth path.
    """
    def __init__(self, gt_data: List[Dict]):
        self.steps = []
        
        for item in gt_data:
            img_data = item['image']
            
            # Handle HF Dataset Serialization (bytes -> PIL)
            if isinstance(img_data, dict):
                if 'bytes' in img_data and img_data['bytes'] is not None:
                    image = Image.open(BytesIO(img_data['bytes'])).convert("RGB")
                else:
                    print(f"[Warning] GTTrajectory: Image dict missing bytes. Using placeholder.")
                    image = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
            elif isinstance(img_data, Image.Image):
                image = img_data
            else:
                print(f"[Warning] GTTrajectory: Unknown image type {type(img_data)}. Using placeholder.")
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
    def __init__(self, input_text: str, start_gt_step: GTStep):
        self.global_question = input_text
        
        # --- State ---
        self.current_gt_step_idx = 0
        self.current_base_image = start_gt_step.image.copy() 
        
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
        
        # [NEW] Visual Path Recording
        self.cursor_path = [tuple(self.cursor_pos)]
        
        # Render the initial frame (Step 0 state)
        self._update_visual_history()

    def _update_visual_history(self):
        """Draws cursor on current base image and appends to history."""
        viz_img = draw_cursor(
            self.current_base_image, 
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
        
        # [NEW] Record Path
        self.cursor_path.append((nx, ny))
        
        self._update_visual_history()

    def advance_to_next_gt_step(self, next_gt_step: GTStep):
        self.current_gt_step_idx += 1
        self.gt_steps_passed += 1
        
        self.current_base_image = next_gt_step.image.copy()
        
        # Reset cursor to center of the NEW image
        w, h = self.current_base_image.size
        self.cursor_pos = [w // 2, h // 2]
        
        # [NEW] Record Reset point as a jump in path (or just new start)
        self.cursor_path.append((w // 2, h // 2))
        
        self._update_visual_history()

# =============================================================================
# RUNNER
# =============================================================================

class Runner:
    def __init__(self):
        print(f"[Runner] Initialized. Capable of dynamic resolution handling.")

    def check_hit(self, cursor_pos, bbox):
        cx, cy = cursor_pos
        x1, y1, x2, y2 = bbox
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def run_trajectory(
        self,
        model,
        processor,
        input_text: str,
        ground_truth_data: List[Dict],
        max_steps: int = 10,
        temperature: float = 1.0
    ) -> Tuple[AgentTrajectory, List[int], List[float]]:
        
        # 1. Init GT Structure
        gt_traj = GTTrajectory(ground_truth_data)
        
        # 2. Init Agent State (Starts at GT Step 0)
        agent_traj = AgentTrajectory(input_text, gt_traj.get_step(0))
        
        # 3. Storage for Trainer
        all_token_ids = []
        all_logprobs = []

        # === ACTION LOOP ===
        for step in range(max_steps):
            agent_traj.step_count += 1
            current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
            
            # 1. Build Prompt
            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": agent_traj.get_current_view()},
                        {"type": "text", "text": f"[Action] Perform a step for the following action: {agent_traj.global_question}\nCurrent Cursor: {agent_traj.cursor_pos}"}
                    ]
                }
            ]
            
            if agent_traj.tools:
                hist_str = "\nAction History:\n" + "\n".join([f"- {t}" for t in agent_traj.tools])
                messages[1]["content"].append({"type": "text", "text": hist_str})

            # 2. Inference
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt],
                images=[agent_traj.get_current_view()],
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                
                if temperature == 0.0:
                    token_id = torch.argmax(logits).item()
                else:
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    token_id = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log_softmax(logits, dim=-1)[token_id].item()

            token_text = processor.decode([token_id], skip_special_tokens=False)
            
            all_token_ids.append(token_id)
            all_logprobs.append(log_prob)
            agent_traj.tools.append(token_text)
            
            # 3. Execution
            token_action_type = get_action_type(token_text)
            
            if token_action_type == "move":
                if token_text in MOVE_DELTAS:
                    dx, dy = MOVE_DELTAS[token_text]
                    agent_traj.move_cursor(dx, dy)
                else:
                    agent_traj.failed = 1 # Invalid
                    break
            
            elif token_action_type in ["click", "scroll", "text", "nav"]:
                if token_action_type != current_gt.action_type:
                    print(f" [Fail] Wrong Type. Expected {current_gt.action_type}, got {token_action_type}")
                    agent_traj.failed = 2 
                    break
                
                if self.check_hit(agent_traj.cursor_pos, current_gt.bbox):
                    # HIT
                    if agent_traj.current_gt_step_idx < gt_traj.total_steps - 1:
                        next_step = gt_traj.get_step(agent_traj.current_gt_step_idx + 1)
                        agent_traj.advance_to_next_gt_step(next_step)
                    else:
                        agent_traj.current_gt_step_idx += 1
                        agent_traj.gt_steps_passed += 1
                else:
                    agent_traj.failed = 3 # Wrong Pos
                    break
            
            elif token_action_type == "end":
                if agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                    agent_traj.failed = 0 # Success
                else:
                    agent_traj.failed = 4 # Premature
                break
            else:
                agent_traj.failed = 1 # Garbage
                break
        
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             if "END_ACTION" not in agent_traj.tools[-1]:
                 agent_traj.failed = 4 # Timeout

        return agent_traj, all_token_ids, all_logprobs