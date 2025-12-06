# src/utils/tokenizer.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .action import ACTION_BASE_EMBEDDING

def add_new_tokens(
    model, 
    processor, 
    base_embedding=ACTION_BASE_EMBEDDING, 
    smart_init=True, 
    use_norm=True
):
    """
    Add custom tokens, resize embeddings, untie weights, and initialize.
    
    Args:
        smart_init (bool): If False, use random initialization (default HF behavior).
                           If True, use semantic averaging of base words.
        use_norm (bool):   If True, apply Spherical Normalization (fix norm collapse).
                           Only active if smart_init is True.
    """
    
    tokenizer = processor.tokenizer
    
    # =========================================================
    # 1. Add tokens to tokenizer
    # =========================================================
    new_tokens = list(base_embedding.keys())
    existing_tokens = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in new_tokens if t not in existing_tokens]
    
    if tokens_to_add:
        tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
        print(f"\n[Tokenizer] Added {len(tokens_to_add)} new tokens. Total vocab: {len(tokenizer)}")
    
    # =========================================================
    # 2. Resize model embeddings
    # =========================================================
    # This initializes new tokens randomly (Mean=0, Std=0.02 usually)
    model.resize_token_embeddings(len(tokenizer))
    
    # =========================================================
    # 3. Untie LM Head from Input Embeddings
    # =========================================================
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    
    if output_embeddings is not None and output_embeddings.weight is input_embeddings.weight:
        print("[Tokenizer] Weight Tying detected. Untying LM Head...")
        new_lm_head_weight = input_embeddings.weight.clone().detach()
        output_embeddings.weight = torch.nn.Parameter(new_lm_head_weight)
        model.config.tie_word_embeddings = False
        print("[Tokenizer] Untying complete.")
    
    # Refresh references after resize/untie
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    # =========================================================
    # 4. Initialization Logic (Ablation Control)
    # =========================================================
    if not smart_init:
        print("[Tokenizer] Mode: RANDOM Initialization (Baseline)")
        # Do nothing. verify_token_embeddings has already initialized them randomly.
        return model, processor

    # If smart_init is True, we calculate anchors
    mode_str = "SPHERICAL MEAN" if use_norm else "ARITHMETIC MEAN"
    print(f"[Tokenizer] Mode: {mode_str} Initialization")
    
    # A. Calculate Target Norm (The "Radius" of the Hypersphere)
    # We want new tokens to have the same magnitude as existing tokens.
    target_norm = 1.0
    if use_norm:
        with torch.no_grad():
            # Calculate average norm of all existing tokens
            target_norm = input_embeddings.weight.norm(dim=1).mean().item()
            print(f"[Tokenizer] Target Norm (Radius): {target_norm:.4f}")

    with torch.no_grad():
        for new_token, base_words in base_embedding.items():
            new_token_id = tokenizer.convert_tokens_to_ids(new_token)
            
            # B. Collect semantic anchors
            anchor_vals = []
            for word in base_words:
                word_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(word_ids) > 0:
                    # Average sub-tokens if a word is split (e.g., "Left" -> ["Le", "ft"])
                    word_emb = input_embeddings.weight[word_ids].mean(dim=0)
                    anchor_vals.append(word_emb)
            
            if anchor_vals:
                # C. Compute Arithmetic Mean (The "Midpoint" inside the sphere)
                # Shape: [Hidden_Dim]
                init_vec = torch.stack(anchor_vals).mean(dim=0)
                
                # D. Apply Spherical Normalization (The Fix)
                if use_norm:
                    current_norm = init_vec.norm(p=2)
                    # Project vector onto the hypersphere surface
                    init_vec = init_vec / (current_norm + 1e-8) * target_norm
                
                # E. Assign to layers
                # Input Layer (Encoder)
                input_embeddings.weight[new_token_id] = init_vec
                
                # Output Layer (Decoder / LM Head)
                if output_embeddings is not None:
                    output_embeddings.weight[new_token_id] = init_vec
            else:
                print(f"[Warning] No anchors found for {new_token}")
    
    print(f"[Tokenizer] Initialization complete.\n")
    return model, processor

def save_model(model, processor, output_dir):
    """Save model and processor."""
    print(f"[Saver] Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[Saver] Saved.")