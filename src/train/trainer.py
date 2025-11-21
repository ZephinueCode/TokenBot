# src/train/trainer.py

import torch
import numpy as np
import os
from typing import Optional, Union, List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from tqdm import tqdm
from PIL import Image

from ..tools.runner import Runner, ToolData, AgentTrajectory
from .reward import batch_compute_rewards
from ..utils.action_logic import MOVE_DELTAS, get_action_type
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.action import ACTION_TOKENS
from ..tools.visual_utils import visualize_trajectory

class ToolGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer tailored for GUI Agents.
    Includes custom generation, reward calculation, and visualization hooks.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: GRPOConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        runner: Optional[Runner] = None,
        max_tool_steps: int = 10,
        beta: float = 0.1,
        **kwargs
    ):
        super().__init__(
            model=model,
            reward_funcs=self._dummy_reward,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs
        )
        
        self.max_tool_steps = max_tool_steps
        self.runner = runner or Runner()
        self.beta = beta
        
        # --- Pre-compute Valid IDs for Masking ---
        valid_action_ids = {
            self.processing_class.tokenizer.convert_tokens_to_ids(t) 
            for t in ACTION_TOKENS
        }
        angle_id = self.processing_class.tokenizer.convert_tokens_to_ids("<")
        im_end_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        self.valid_ids_set = valid_action_ids.union({angle_id, im_end_id})
        self.valid_ids_list = list(self.valid_ids_set) 
        
        # --- Embedding Freeze & Untie Logic ---
        print(f"\n[ToolGRPOTrainer] Configuring Model Layers...")
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None:
                embeddings.requires_grad_(False)
                print(" -> Input embeddings FROZEN.")

        if self.model.config.tie_word_embeddings:
            print(" -> Weight Tying detected. Untying LM Head...")
            if hasattr(self.model, "get_output_embeddings"):
                lm_head = self.model.get_output_embeddings()
                if lm_head.weight is embeddings.weight:
                    new_weight = embeddings.weight.clone().detach()
                    lm_head.weight = torch.nn.Parameter(new_weight)
                    self.model.config.tie_word_embeddings = False
                lm_head.requires_grad_(True)
                print(" -> LM Head UNTIED and set to TRAINABLE.")
        else:
            if hasattr(self.model, "get_output_embeddings"):
                self.model.get_output_embeddings().requires_grad_(True)

    def _dummy_reward(self, prompts, completions, **kwargs):
        return [0.0] * len(prompts)

    def _prepare_inputs(self, inputs: Union[dict, list]) -> dict:
        if isinstance(inputs, dict):
            if "question" in inputs and isinstance(inputs["question"], list):
                batch_size = len(inputs["question"])
                inputs_list = [
                    {
                        "question": inputs["question"][i],
                        "image": inputs["image"][i],
                        "ground_truth_traj": inputs["ground_truth_traj"][i],
                    }
                    for i in range(batch_size)
                ]
            else:
                if "per_token_logps" in inputs: return inputs
                raise ValueError("Input dict must contain lists for batching.")
        else:
            inputs_list = inputs
            
        return self._generate_and_score_completions(inputs_list)

    def _generate_and_score_completions(self, inputs: list[dict]) -> dict:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_size = len(inputs)
        
        all_trajectories = []
        all_token_ids = []
        all_old_logprobs = []
        
        if mode == "train":
            print(f"\n[Gen] Generating trajectories for batch of {batch_size}...", end="", flush=True)

        for sample in inputs:
            question_text = sample['question']
            
            traj, token_ids, logprobs = self.runner.run_trajectory(
                model=self.model,
                processor=self.processing_class,
                input_text=question_text,
                ground_truth_data=sample['ground_truth_traj'],
                max_steps=self.max_tool_steps,
                temperature=self.args.temperature if mode == "train" else 0.0
            )
            all_trajectories.append(traj)
            all_token_ids.append(token_ids)
            all_old_logprobs.append(logprobs)

        if mode == "train":
            print(" Done.")

        # --- Compute Rewards ---
        current_epoch = self.state.epoch if self.state.epoch is not None else 0.0
        rewards = batch_compute_rewards(all_trajectories, epoch=current_epoch)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        all_rewards_gathered = self.accelerator.gather(rewards_tensor)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # --- Recompute Logprobs (Replay) ---
        all_current_logprobs = self._recompute_logprobs(all_trajectories, all_token_ids)
        
        # --- Padding & Batching ---
        valid_token_ids = [ids for ids in all_token_ids if ids]
        max_len = max((len(ids) for ids in valid_token_ids), default=1)
        
        pad_ids, pad_mask, pad_curr, pad_old = [], [], [], []
        pad_token_id = self.processing_class.tokenizer.pad_token_id
        
        for i in range(batch_size):
            # Handle skipped (OOM) or empty trajectory
            if not all_token_ids[i] or len(all_current_logprobs[i]) == 0:
                pad_ids.append(torch.tensor([pad_token_id], device=device, dtype=torch.long))
                pad_mask.append(torch.tensor([0], device=device, dtype=torch.long))
                pad_curr.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                pad_old.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                continue

            length = len(all_token_ids[i])
            diff = max_len - length
            if diff < 0: diff = 0

            ids = all_token_ids[i] + [pad_token_id] * diff
            mask = [1] * length + [0] * diff
            pad_ids.append(torch.tensor(ids, device=device, dtype=torch.long))
            pad_mask.append(torch.tensor(mask, device=device, dtype=torch.long))
            
            curr = all_current_logprobs[i]
            old = torch.tensor(all_old_logprobs[i], device=device, dtype=torch.float32)
            
            if diff > 0:
                curr = torch.cat([curr, torch.zeros(diff, device=device, dtype=curr.dtype)])
                old = torch.cat([old, torch.zeros(diff, device=device, dtype=old.dtype)])
            
            pad_curr.append(curr)
            pad_old.append(old)
            
        return {
            "raw_rewards": rewards_tensor, 
            "completion_mask": torch.stack(pad_mask),
            "advantages": advantages.detach(),
            "per_token_logps": torch.stack(pad_curr),
            "old_token_logps": torch.stack(pad_old),
            "completion_ids": torch.stack(pad_ids)
        }

    def _recompute_logprobs(self, trajectories: List[AgentTrajectory], token_ids_list: List[List[int]]):
        """
        Re-run forward pass for gradient calculation (Replay).
        [MODIFIED] Includes OOM Retry Logic: Try once, if fail -> empty_cache -> try again.
        """
        all_logprobs = []
        
        for idx, (traj, token_ids) in enumerate(zip(trajectories, token_ids_list)):
            if not token_ids:
                all_logprobs.append(torch.tensor([], device=self.model.device))
                continue
            
            # [NEW] Retry loop for OOM
            for attempt in range(2):
                try:
                    traj_logprobs = []
                    history_tokens = [] 
                    init_img = traj.images[0]
                    
                    if idx == 0 and len(token_ids) > 0:
                         print(f"\n--- Replay Debug (Sample 0, {len(token_ids)} steps, Attempt {attempt+1}) ---")

                    for step_i, target_id in enumerate(token_ids):
                        step_image = traj.images[step_i]
                        
                        full_generated_text = traj.tools[step_i]
                        target_token_str = self.processing_class.decode([target_id], skip_special_tokens=False)
                        
                        if full_generated_text.endswith(target_token_str):
                            assistant_prefix = full_generated_text[:-len(target_token_str)]
                        else:
                            assistant_prefix = full_generated_text.replace(target_token_str, "")

                        messages = [
                            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": step_image},
                                    {"type": "text", "text": f"[Action] {traj.global_question}\n"}
                                ]
                            }
                        ]
                        
                        if history_tokens:
                            hist_str = "\nAction History:\n" + "\n".join([f"- {t}" for t in history_tokens])
                            messages[1]["content"].append({"type": "text", "text": hist_str})
                        
                        prompt_text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        final_input_text = prompt_text + assistant_prefix
                        
                        inputs = self.processing_class(
                            text=[final_input_text], 
                            images=[step_image], 
                            return_tensors="pt", 
                            padding=True
                        ).to(self.model.device)
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits[0, -1, :]

                        # Apply Masking
                        if "Action: " in final_input_text[-20:]:
                            mask = torch.full_like(logits, -1e9)
                            mask[self.valid_ids_list] = 0.0
                            logits = logits + mask
                        
                        lp = torch.log_softmax(logits, dim=-1)[target_id]
                        traj_logprobs.append(lp)
                        
                        if idx == 0 and attempt == 0:
                             with torch.no_grad():
                                 pred_id = torch.argmax(logits).item()
                                 pred_str = self.processing_class.decode([pred_id], skip_special_tokens=False)
                                 debug_context = assistant_prefix[-20:].replace('\n', '\\n')
                                 print(f" Step {step_i}:")
                                 print(f"   Context (End): '...{debug_context}'")
                                 print(f"   Target: '{target_token_str}' | Pred: '{pred_str}'")
                                 print(f"   Logprob: {lp.item():.4f}")

                        history_tokens.append(full_generated_text)

                    # Success!
                    all_logprobs.append(torch.stack(traj_logprobs))
                    break # Exit retry loop

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if attempt == 0:
                            print(f"\n[Trainer] OOM for sample {idx}. Clearing cache and retrying...")
                            continue # Retry
                        else:
                            print(f"\n[Trainer] OOM again for sample {idx}. Skipping.")
                            all_logprobs.append(torch.tensor([], device=self.model.device))
                            break # Give up
                    else:
                        raise e # Re-raise other errors
            
            torch.cuda.empty_cache()
            
        return all_logprobs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs: raise ValueError("Not supported")
        
        per_token_logps = inputs["per_token_logps"]
        old_token_logps = inputs["old_token_logps"]
        mask = inputs["completion_mask"]
        advantages = inputs["advantages"]

        raw_rewards = inputs.get("raw_rewards", torch.tensor(0.0))
        current_raw_reward_mean = raw_rewards.mean().item()

        traj_logps = (per_token_logps * mask).sum(dim=1)
        traj_old_logps = (old_token_logps * mask).sum(dim=1)
        
        pg_loss = -(advantages * traj_logps).mean()
        kl_penalty = (traj_logps - traj_old_logps).pow(2).mean()
        loss = pg_loss + self.beta * kl_penalty
        
        if self.model.training:
            self.log({
                "loss/total": loss.item(),
                "loss/pg": pg_loss.item(),
                "loss/kl": kl_penalty.item(),
                "reward/raw_mean": current_raw_reward_mean,
                "reward/advantage_mean": advantages.mean().item() 
            })
            
        return loss

    # ... (Evaluation loop remains unchanged) ...
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        current_step = self.state.global_step
        vis_save_dir = f"./results/eval_step_{current_step}"
        os.makedirs(vis_save_dir, exist_ok=True)
        print(f"\n[{metric_key_prefix}] Starting Evaluation Loop...")
        print(f"[{metric_key_prefix}] Saving visualizations to: {vis_save_dir}")
        all_rewards = []
        success_count = 0
        total_steps = 0
        failed_counts = {1:0, 2:0, 3:0, 4:0}
        global_sample_idx = 0
        for step, batch_inputs in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating"):
            if isinstance(batch_inputs, dict):
                bs = len(batch_inputs["question"])
                inputs_list = [{
                        "question": batch_inputs["question"][i],
                        "image": batch_inputs["image"][i],
                        "ground_truth_traj": batch_inputs["ground_truth_traj"][i],
                    } for i in range(bs)]
            else: inputs_list = batch_inputs
            for sample in inputs_list:
                question_text = sample['question']
                traj, _, _ = self.runner.run_trajectory(
                    model=model,
                    processor=self.processing_class,
                    input_text=question_text,
                    ground_truth_data=sample['ground_truth_traj'],
                    max_steps=self.max_tool_steps,
                    temperature=0.0 
                )
                reward = batch_compute_rewards([traj])[0]
                all_rewards.append(reward)
                total_steps += traj.step_count
                is_success = (traj.failed == 0)
                if is_success: success_count += 1
                else:
                    if traj.failed in failed_counts: failed_counts[traj.failed] += 1
                try:
                    raw_img = sample['image']
                    if isinstance(raw_img, np.ndarray): base_img = Image.fromarray(raw_img).convert("RGB")
                    else: base_img = raw_img.convert("RGB")
                    gt_data = sample['ground_truth_traj']
                    gt_bbox = gt_data[0]['bbox'] if gt_data else None
                    w, h = raw_img.size
                    if gt_bbox and all(0.0 <= c <= 1.0 for c in gt_bbox):
                        gt_bbox = [gt_bbox[0]*w, gt_bbox[1]*h, gt_bbox[2]*w, gt_bbox[3]*h]
                    
                    vis_img = visualize_trajectory(
                        base_image=base_img,
                        cursor_path=traj.cursor_path,
                        actions=traj.tools,
                        gt_bbox=gt_bbox,
                        success=is_success,
                        instruction=traj.global_question
                    )
                    status_str = "PASS" if is_success else "FAIL"
                    filename = f"id_{global_sample_idx:04d}_{status_str}_rew{int(reward)}.png"
                    vis_img.save(os.path.join(vis_save_dir, filename))
                except Exception as e: print(f"[Eval] Visualization failed: {e}")
                global_sample_idx += 1
                torch.cuda.empty_cache()

        num_samples = len(all_rewards)
        if num_samples == 0: return EvalLoopOutput(predictions=None, label_ids=None, metrics={}, num_samples=0)
        avg_reward = sum(all_rewards) / num_samples
        success_rate = success_count / num_samples
        avg_steps = total_steps / num_samples
        metrics = {
            f"{metric_key_prefix}_reward": avg_reward,
            f"{metric_key_prefix}_success_rate": success_rate,
            f"{metric_key_prefix}_avg_steps": avg_steps,
            f"{metric_key_prefix}_fail_invalid": failed_counts[1] / num_samples,
            f"{metric_key_prefix}_fail_miss": failed_counts[3] / num_samples, 
            f"{metric_key_prefix}_fail_timeout": failed_counts[4] / num_samples,
        }
        print(f"\n{'='*40}")
        print(f"EVALUATION RESULTS ({num_samples} samples)")
        print(f"Avg Reward:   {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Avg Steps:    {avg_steps:.2f}")
        print(f"{'='*40}\n")
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)