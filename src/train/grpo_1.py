# src/train/grpo_1.py

import torch
import os
import glob
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from trl import GRPOConfig

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.sft_screenspot_pro import prepare_grpo1_dataset, train_eval_split
from ..tools.runner import Runner
from .trainer import ToolGRPOTrainer
from ..utils.action import ACTION_TOKENS 

def get_last_checkpoint(output_dir):
    """
    Helper to find the latest checkpoint directory.
    Returns None if no checkpoints found.
    """
    if not os.path.exists(output_dir):
        return None
    
    # Check for subdirectories named 'checkpoint-*'
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number (assuming format checkpoint-X)
    try:
        # Extract number from path and sort
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return latest_ckpt
    except ValueError:
        return None

def run_grpo_1():
    print(f"[GRPO-1] Initializing Pipeline...")
    
    # 1. Load Post-SFT Model
    model_path = HP.SFT_OUTPUT_PATH 
    print(f"[GRPO-1] Loading model from {model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model.enable_input_require_grads()

    # 2. Dataset
    full_dataset = prepare_grpo1_dataset(HP.GRPO1_DATA_PATH, size=HP.GRPO_DATASET_SIZE)
    train_data, eval_data = train_eval_split(full_dataset, eval_ratio=0.05)
    print(f"[GRPO-1] Data: {len(train_data)} Train, {len(eval_data)} Eval")
    
    # 3. Runner
    runner = Runner()
    
    # 4. Config (Output path is HP.GRPO1_OUTPUT_PATH)
    training_args = GRPOConfig(
        output_dir=HP.GRPO1_OUTPUT_PATH,
        learning_rate=HP.GRPO1_LEARN_RATE,
        num_train_epochs=HP.GRPO1_NUM_EPOCHS,
        per_device_train_batch_size=HP.GRPO1_BATCH_SIZE,
        gradient_accumulation_steps=HP.GRPO1_GRAD_ACCUM_STEPS,
        num_generations=HP.GRPO1_NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=512, 
        logging_steps=HP.GRPO1_LOGGING_STEPS,
        save_steps=HP.GRPO1_SAVE_STEPS,
        max_grad_norm=1.0,
        temperature=HP.GRPO1_TEMPERATURE,
        optim="adamw_bnb_8bit",
        report_to="tensorboard",
        remove_unused_columns=False,
        warmup_ratio=0.05,
        beta=0.02, 
        
        do_eval=True,
        eval_strategy="steps",
        eval_steps=HP.GRPO1_EVAL_STEPS,
        eval_on_start=False,
        save_total_limit=HP.GRPO1_MAX_CHECKPOINTS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward",
        greater_is_better=True,
        gradient_checkpointing=True
    )
    
    # 5. Trainer
    trainer = ToolGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
        runner=runner,
        max_tool_steps=10,
        beta=0.1
    )
    
    # =========================================================================
    # CRITICAL: Re-apply Freezing Strategy (NO HOOKS, matching SFT)
    # =========================================================================
    print("[GRPO-1] Re-applying Freezing Strategy...")
    
    # [UPDATE] Remove the hook logic to match your SFT strategy
    # Freeze Vision, Unfreeze Full LLM
    
    trainable = 0
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable += param.numel()
            
    print(f"[GRPO-1] Strategy Applied: Vision Frozen, Full LLM Unfrozen.")
    print(f"[GRPO-1] Trainable Params: {trainable:,}")
    # =========================================================================
    
    # 6. Train (With Resume Logic)
    print("[GRPO-1] Checking for existing checkpoints...")
    last_checkpoint = get_last_checkpoint(HP.GRPO1_OUTPUT_PATH)
    
    if last_checkpoint:
        print(f"[GRPO-1] Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("[GRPO-1] No checkpoint found. Starting fresh training...")
        trainer.train()
    
    # 7. Save
    print(f"[GRPO-1] Saving final model to {HP.GRPO1_OUTPUT_PATH}")
    trainer.save_model(HP.GRPO1_OUTPUT_PATH)
    trainer.save_state()
    processor.save_pretrained(HP.GRPO1_OUTPUT_PATH)