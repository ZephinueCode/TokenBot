# src/__main__.py

import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .utils.parameters import HYPERPARAMS as HP
from .utils.tokenizer import add_new_tokens, save_model
from .utils.action import ACTION_BASE_EMBEDDING

# Import Training Phases
from .train.sft import run_sft as run_sft_phase1
from .train.sft2 import run_sft_screenspot as run_sft_phase2
from .train.sft3 import run_sft_screenspot_pro as run_sft_phase3

if __name__ == "__main__":
    
    # =========================================================================
    # PHASE 0: Model Initialization
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 0: Model Initialization")
    print("="*60 + "\n")
    
    if os.path.exists(HP.INIT_MODEL_PATH):
        print(f"[Main] Found initialized model at {HP.INIT_MODEL_PATH}. Skipping.")
    else:
        print(f"[Main] Loading base model from: {HP.BASE_MODEL_PATH}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            HP.BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(HP.BASE_MODEL_PATH, trust_remote_code=True)
        
        print("[Main] Injecting GUI Action Tokens...")
        model, processor = add_new_tokens(model, processor, ACTION_BASE_EMBEDDING)
        
        save_model(model, processor, HP.INIT_MODEL_PATH)
        print(f"PHASE 0 COMPLETE. Saved to: {HP.INIT_MODEL_PATH}")
    
    # =========================================================================
    # PHASE 1: SFT - Semantic Injection (Synthetic Data)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: SFT - Semantic Injection")
    print("Action: Training on JSONL data & Synthetic Blanks.")
    print("="*60 + "\n")
    
    # Check if Phase 1 is already done
    if not os.path.exists(HP.SFT_OUTPUT_PATH):
        run_sft_phase1()
    else:
        print("[Main] Phase 1 output found. Skipping.")
    
    # =========================================================================
    # PHASE 2: SFT - Visual Grounding (Legacy ScreenSpot)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: SFT - Visual Grounding (Basic)")
    print("Action: Training on ScreenSpot shortest-path trajectories.")
    print("="*60 + "\n")
    
    if not os.path.exists(HP.SFT_2_OUTPUT_PATH):
        run_sft_phase2()
    else:
        print("[Main] Phase 2 output found. Skipping.")

    '''
    # =========================================================================
    # PHASE 3: SFT - Visual Grounding Pro (ScreenSpot Pro)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: SFT - Visual Grounding Pro")
    print("Action: Training on ScreenSpot Pro local dataset.")
    print("="*60 + "\n")

    # run_sft_phase3()
    
    print("\n" + "="*60)
    # The final model is now the output of Phase 3
    final_path = getattr(HP, "SFT_3_OUTPUT_PATH", "./checkpoints/sft_phase3")
    print(f"PIPELINE COMPLETE. Final Model: {final_path}")
    print("="*60 + "\n")
    '''