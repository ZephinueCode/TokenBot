# src/eval/baseline_grounding.py

import torch
import base64
import re
import os
import json
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
from openai import OpenAI

from ..utils.parameters import HYPERPARAMS as HP
from ..tools.runner import Runner, AgentTrajectory, GTTrajectory
from ..tools.visual_utils import draw_cursor
from ..utils.prompts import BASELINE_GROUNDING_PROMPT

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def encode_image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# =============================================================================
# RUNNER CLASS
# =============================================================================

class BaselineGroundingRunner(Runner):
    """
    A baseline agent that follows the 'Computer Use' JSON prompt format.
    It parses JSON actions (e.g., {"action": "left_click", "coordinate": [x, y]}).
    """

    def __init__(self):
        print(f"[BaselineGroundingRunner] Initializing API Client...")
        print(f" - Base URL: {HP.VLM_BASE_URL}")
        print(f" - Model: {HP.BASELINE_MODEL_NAME}")
        
        self.client = OpenAI(
            api_key=HP.VLM_API_KEY,
            base_url=HP.VLM_BASE_URL
        )

    def parse_response(self, content: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parses response format:
        Reasoning: ...
        Action: ```json {...} ``` or just {...}
        
        Returns: (reasoning, action_name, params_dict)
        """
        content = content.strip()
        reasoning = ""
        action_json_str = ""

        # 1. Split Reasoning and Action
        if "Action:" in content:
            parts = content.split("Action:")
            reasoning = parts[0].replace("Reasoning:", "").strip()
            action_json_str = parts[1].strip()
        else:
            # Fallback: try to find the last JSON block
            reasoning = content
            action_json_str = content

        # 2. Extract JSON string
        # Handle markdown code blocks if present
        if "```json" in action_json_str:
            match = re.search(r"```json(.*?)```", action_json_str, re.DOTALL)
            if match:
                action_json_str = match.group(1).strip()
        elif "```" in action_json_str:
             match = re.search(r"```(.*?)```", action_json_str, re.DOTALL)
             if match:
                action_json_str = match.group(1).strip()

        # 3. Parse JSON
        try:
            # Clean up potentially messy string
            action_json_str = action_json_str.strip()
            data = json.loads(action_json_str)
            
            if isinstance(data, dict) and "action" in data:
                action_name = data["action"]
                # Remove 'action' key to return the rest as parameters
                params = {k: v for k, v in data.items() if k != "action"}
                return reasoning, action_name, params
                
        except json.JSONDecodeError:
            pass
            
        return reasoning, None, {}

    def run_trajectory(
        self,
        input_text: str,
        ground_truth_data: List[dict],
        max_steps: int = 10
    ) -> Tuple[AgentTrajectory, None, None]:
        
        # 1. Init State
        gt_traj = GTTrajectory(ground_truth_data)
        
        # [Guard] Check for empty GT
        if gt_traj.total_steps == 0:
            print("[Error] Ground Truth data is empty.")
            dummy = AgentTrajectory(input_text, None, 0)
            dummy.failed = 4
            return dummy, None, None

        agent_traj = AgentTrajectory(input_text, gt_traj.get_step(0), total_gt_steps=gt_traj.total_steps)
        
        print(f"\n=== AGENT TASK: {input_text} ===")

        # === ACTION LOOP ===
        for step in range(max_steps):
            agent_traj.step_count += 1
            
            # [Guard] Safe GT retrieval
            current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
            if current_gt is None:
                print(" -> [Fail] GT Index out of bounds.")
                agent_traj.failed = 4
                break

            # 2. Prepare View and Resolution
            current_view = agent_traj.get_current_view()
            W, H = current_view.size
            base64_img = encode_image_to_base64(current_view)
            
            # [Debug] Save monitoring image
            try:
                os.makedirs("./results", exist_ok=True) 
                current_view.save("./results/current_grounding.png")
            except Exception: pass

            # 3. Build Prompt (Inject Resolution)
            system_prompt_with_res = f"{BASELINE_GROUNDING_PROMPT}\n\n# CURRENT STATE\nThe resolution of the image is {W}x{H}."
            
            user_content = f"Instruction: {agent_traj.global_question}"
            
            messages = [
                {"role": "system", "content": system_prompt_with_res},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                        {"type": "text", "text": user_content}
                    ]
                }
            ]

            # 4. Call API
            try:
                completion = self.client.chat.completions.create(
                    model=HP.BASELINE_MODEL_NAME,
                    messages=messages,
                    temperature=0.0, 
                    max_tokens=256
                )
                response_text = completion.choices[0].message.content
            except Exception as e:
                print(f"[API Error] Step {step}: {e}")
                agent_traj.failed = 1
                break

            # 5. Parse Output
            reasoning, action_name, params = self.parse_response(response_text)
            
            print(f"\n[Step {step+1}]")
            print(f"  Thinking: {reasoning[:100]}...")
            print(f"  Action:   {action_name} {params}")
            
            agent_traj.tools.append(f"{action_name} {params}")

            if not action_name:
                print(f"  -> [Fail] Invalid JSON format: {response_text[:50]}")
                agent_traj.failed = 1
                break

            # 6. Execute Logic
            cx, cy = -1, -1
            
            # Extract coordinates if "coordinate" exists in params
            if "coordinate" in params:
                coords = params["coordinate"]
                if isinstance(coords, list) and len(coords) == 2:
                    cx, cy = int(coords[0]), int(coords[1])
                    
                    # Update Cursor Position (Virtual Move)
                    dx = cx - agent_traj.cursor_pos[0]
                    dy = cy - agent_traj.cursor_pos[1]
                    agent_traj.move_cursor(dx, dy)

            # --- Interaction Logic ---
            # Qwen Agent specific action names
            click_actions = ["left_click", "right_click", "middle_click", "double_click", "left_click_drag"]
            
            if action_name in click_actions:
                # Verify Hit
                if self.check_hit(agent_traj.cursor_pos, current_gt.bbox, current_view.size):
                    print(" -> Hit! Target reached.")
                    
                    if agent_traj.current_gt_step_idx < gt_traj.total_steps - 1:
                        next_step = gt_traj.get_step(agent_traj.current_gt_step_idx + 1)
                        agent_traj.advance_to_next_gt_step(next_step)
                    else:
                        agent_traj.current_gt_step_idx += 1
                        agent_traj.gt_steps_passed += 1
                        print(" -> Task Success.")
                        agent_traj.failed = 0
                        break
                else:
                    print(f" -> Miss! Clicked at ({cx}, {cy}).")
                    agent_traj.failed = 3 
                    break 

            # --- Other Actions ---
            elif action_name == "type":
                pass 

            elif action_name == "terminate":
                status = params.get("status", "failure")
                if status == "success" and agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                     agent_traj.failed = 0
                else:
                     agent_traj.failed = 4
                break

            elif action_name == "wait":
                pass 

            elif action_name == "mouse_move":
                # Just move, already handled above by coordinate extraction
                pass

            else:
                print(f" -> [Warn] Unknown command: {action_name}")
        
        # Timeout Check
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             if agent_traj.current_gt_step_idx < gt_traj.total_steps:
                 print(f" [Runner] Max steps ({max_steps}) reached. Timeout.")
                 agent_traj.failed = 4

        return agent_traj, None, None