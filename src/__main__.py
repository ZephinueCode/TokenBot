# src/__main__.py

import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .utils.parameters import HYPERPARAMS as HP
from .utils.tokenizer import add_new_tokens, save_model
from .utils.action import ACTION_BASE_EMBEDDING
from .train.grpo_1 import run_grpo_1

# Import Training Function
from .train.sft import run_sft

if __name__ == "__main__":
    
    # =========================================================================
    # PHASE 0: Initialization (Add GUI Tokens & Semantic Anchoring)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 0: Model Initialization")
    print("Action: Adding GUI tokens to vocabulary and resizing embeddings.")
    print("="*60 + "\n")
    
    # 1. Configuration
    MODEL_PATH = HP.BASE_MODEL_PATH
    OUTPUT_PATH = HP.INIT_MODEL_PATH
    
    # Check if initialization is already done to save time
    if os.path.exists(OUTPUT_PATH):
        print(f"[Main] Found initialized model at {OUTPUT_PATH}. Skipping Phase 0.")
    else:
        # 2. Load Base Model & Processor
        print(f"[Main] Loading base model from: {MODEL_PATH}")
        try:
            # Note: Qwen3VL typically requires trust_remote_code=True
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[Error] Failed to load model. Please check the path in src/utils/parameters.py.\nDetails: {e}")
            exit(1)
        
        # 3. Add Tokens & Smart Init
        print("[Main] Injecting GUI Action Tokens...")
        model, processor = add_new_tokens(model, processor, ACTION_BASE_EMBEDDING)
        
        # 4. Save Initialized Model
        save_model(model, processor, OUTPUT_PATH)
        print("\n" + "="*60)
        print(f"PHASE 0 COMPLETE. Initialized model saved to:\n{OUTPUT_PATH}")
        print("="*60 + "\n")

    # =========================================================================
    # PHASE 1: SFT 1 (Semantic Injection - Blank Image Training)
    # =========================================================================
    '''
    print("\n" + "="*60)
    print("PHASE 1: SFT - Semantic Injection")
    print("Action: Training Action Token embeddings on JSONL data (Blank Images).")
    print("="*60 + "\n")
    
    # Run the training pipeline
    # Input: INIT_MODEL_PATH -> Output: SFT_1_OUTPUT_PATH
    run_sft()
    
    print("\n" + "="*60)
    print(f"PHASE 1 COMPLETE. Model saved to:\n{HP.SFT_OUTPUT_PATH}")
    print("="*60 + "\n")
    '''
    
    # =========================================================================
    # PHASE 2: GRPO 1 (Visual Grounding)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: GRPO 1 - Visual Grounding")
    print("Action: Reinforcement Learning on ScreenSpot to learn cursor control.")
    print("="*60 + "\n")
    
    run_grpo_1()
    
    print("\n" + "="*60)
    print(f"PHASE 2 COMPLETE. Model saved to:\n{HP.GRPO1_OUTPUT_PATH}")
    print("="*60 + "\n")
    