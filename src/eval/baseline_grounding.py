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
from ..utils.prompts import BASELINE_GROUNDING_PROMPT as TOOLS_DEFINITION

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
    A baseline agent that follows the 'Computer Use' XML/JSON prompt format.
    It forces the input image to 1000x1000 and maps coordinates back to original size.
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
        <tool_call>
        {"name": "computer_use", "arguments": {"action": "...", "coordinate": [x, y]}}
        </tool_call>
        """
        content = content.strip()
        reasoning = content 
        action_type = None
        arguments = {}

        # 1. Regex to find <tool_call> content
        match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        
        if match:
            reasoning = content[:match.start()].strip()
            json_str = match.group(1).strip()
            
            try:
                data = json.loads(json_str)
                if data.get("name") == "computer_use":
                    args = data.get("arguments", {})
                    action_type = args.get("action")
                    arguments = args
                    
            except json.JSONDecodeError:
                print(f"[Parse Error] Invalid JSON in tool_call: {json_str}")

        return reasoning, action_type, arguments

    def run_trajectory(
        self,
        input_text: str,
        ground_truth_data: List[dict],
        max_steps: int = 10
    ) -> Tuple[AgentTrajectory, None, None]:
        
        # 1. Init State
        gt_traj = GTTrajectory(ground_truth_data)
        
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
            
            current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
            if current_gt is None:
                print(" -> [Fail] GT Index out of bounds.")
                agent_traj.failed = 4
                break

            # 2. Prepare View (Resize to 1000x1000)
            current_view = agent_traj.get_current_view()
            
            # Store Original Dimensions for mapping back later
            orig_w, orig_h = current_view.size 
            
            # [CRITICAL] Resize to 1000x1000 for the model
            target_size = (1000, 1000)
            resized_view = current_view.resize(target_size, Image.Resampling.LANCZOS)
            
            base64_img = encode_image_to_base64(resized_view)

            # 3. Build Prompt (Fixed 1000x1000 context)
            # The model thinks it is working on a 1000x1000 screen
            system_prompt = TOOLS_DEFINITION.replace("{WIDTH}", "1000").replace("{HEIGHT}", "1000")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                        {"type": "text", "text": f"User Query: {agent_traj.global_question}"}
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
            reasoning, action_type, params = self.parse_response(response_text)
            
            print(f"\n[Step {step+1}]")
            print(f"  Thinking: {reasoning[:100]}...")
            print(f"  Action:   {action_type} {params}")
            
            agent_traj.tools.append(f"{action_type} {params}")

            if not action_type:
                # print(f"  -> [Fail] No valid tool_call found: {response_text[:50]}")
                agent_traj.failed = 1
                break

            # 6. Execute Logic & Coordinate De-Normalization
            cx_abs, cy_abs = -1, -1
            
            if "coordinate" in params:
                coords = params["coordinate"]
                if isinstance(coords, list) and len(coords) == 2:
                    norm_x, norm_y = coords[0], coords[1]
                    
                    # [CRITICAL] Map 1000x1000 back to Original Resolution
                    cx_abs = int((norm_x / 1000.0) * orig_w)
                    cy_abs = int((norm_y / 1000.0) * orig_h)
                    print(f"  -> Mapped Coordinates: ({cx_abs}, {cy_abs}) in Original Size ({orig_w}, {orig_h})")
                    
                    # Update Virtual Cursor Position (Delta based on real dimensions)
                    dx = cx_abs - agent_traj.cursor_pos[0]
                    dy = cy_abs - agent_traj.cursor_pos[1]
                    agent_traj.move_cursor(dx, dy)

            # --- Dispatch Actions ---
            if action_type == "mouse_move":
                pass

            elif action_type == "left_click":
                # Verify Hit using the absolute coordinates and original size
                # Note: We use `current_view.size` (original), not `resized_view`
                if self.check_hit(agent_traj.cursor_pos, current_gt.bbox, (orig_w, orig_h)):
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
                    print(f" -> Miss! Clicked at ({cx_abs}, {cy_abs}).")
                    agent_traj.failed = 3 
                    break 

            elif action_type == "terminate":
                status = params.get("status", "failure")
                if status == "success" and agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                     agent_traj.failed = 0
                else:
                     agent_traj.failed = 4
                break

            else:
                print(f" -> [Warn] Unknown action type: {action_type}")
                agent_traj.failed = 1
                break
        
        # Timeout Check
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             if agent_traj.current_gt_step_idx < gt_traj.total_steps:
                 print(f" [Runner] Max steps ({max_steps}) reached. Timeout.")
                 agent_traj.failed = 4

        return agent_traj, None, None