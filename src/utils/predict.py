import torch
import argparse
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor,
)
# This file contains code for prediction

def predict_next(
    model,
    processor,
    messages,
    constrain_tokens=None,
    temperature=1.0,
    top_k=None,
):
    """
    Predict next token, optionally sampling only from constrain_tokens.
    
    Args:
        model: The language model
        processor: The processor
        messages: List of message dicts with 'role' and 'content'
        constrain_tokens: List of allowed tokens (default: None, no constraints)
        temperature: Sampling temperature
        top_k: If set, only sample from top k tokens (default: None)
        
    Returns:
        predicted_token: The predicted token string
        logits_dict: Dictionary mapping tokens to their logits
        probs_dict: Dictionary mapping tokens to their probabilities
    """
    
    tokenizer = processor.tokenizer
    
    # Prepare input text using chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the next token (last position in sequence)
        next_token_logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)
    
    # Apply temperature
    scaled_logits = next_token_logits / temperature
    
    # Apply constraints if specified
    if constrain_tokens is not None:
        # Convert token strings to token IDs
        constrain_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in constrain_tokens]
        constrain_token_ids_tensor = torch.tensor(constrain_token_ids, device=model.device)
        
        # Create a mask for allowed tokens
        mask = torch.full_like(scaled_logits, float('-inf'))
        mask[constrain_token_ids_tensor] = 0
        scaled_logits = scaled_logits + mask
        
        # Extract constrained logits for output
        constrained_logits = next_token_logits[constrain_token_ids_tensor]
        tokens_to_report = constrain_tokens
        token_ids_to_report = constrain_token_ids_tensor
    else:
        # No constraints - use all tokens
        constrained_logits = scaled_logits
        
        # Apply top_k filtering if specified
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(scaled_logits, k=min(top_k, scaled_logits.size(-1)))
            
            # Create mask for top_k
            mask = torch.full_like(scaled_logits, float('-inf'))
            mask[top_k_indices] = 0
            scaled_logits = scaled_logits + mask
            
            # For reporting purposes
            constrained_logits = next_token_logits[top_k_indices]
            tokens_to_report = [tokenizer.decode([idx]) for idx in top_k_indices.cpu().tolist()]
            token_ids_to_report = top_k_indices
        else:
            # Report all tokens (this could be huge, so limit to top 50 for dict output)
            top_50_logits, top_50_indices = torch.topk(next_token_logits, k=50)
            constrained_logits = top_50_logits
            tokens_to_report = [tokenizer.decode([idx]) for idx in top_50_indices.cpu().tolist()]
            token_ids_to_report = top_50_indices
    
    # Calculate probabilities
    probabilities = torch.softmax(scaled_logits, dim=0)
    
    # Sample from the probability distribution
    sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
    predicted_token = tokenizer.decode([sampled_idx])
    
    # Get the actual probabilities for the tokens we're reporting
    reported_probs = probabilities[token_ids_to_report]
    
    # Convert tensors to dictionaries for easy access
    logits_dict = {token: logits.item() 
                   for token, logits in zip(tokens_to_report, constrained_logits.cpu())}
    probs_dict = {token: prob.item() 
                  for token, prob in zip(tokens_to_report, reported_probs.cpu())}
    
    return predicted_token, logits_dict, probs_dict


def complete_text(
    model,
    processor,
    messages,
    max_new_tokens=100,
    temperature=1.0,
):
    """
    Generate a completion for the given messages.
    
    Args:
        model: The language model
        processor: The processor
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        completion: The generated text completion
    """
    
    # Prepare input text using chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)
    
    # Generate completion
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    
    # Decode the generated tokens (excluding the input)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    completion = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return completion


# Testing area for singular prediction
if __name__ == "__main__":
    import base64
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Test model prediction")
    parser.add_argument("-c", "--complete", action="store_true", 
                        help="Complete the text instead of predicting next token")
    parser.add_argument("-n", "--next", action="store_true",
                        help="Predict next token (default behavior)")
    parser.add_argument("-i", "--image", type=str, default="./image.png",
                        help="Path to image file (default: ./image.png)")
    
    args = parser.parse_args()
    
    # Default to next token prediction if neither flag is set
    if not args.complete and not args.next:
        args.next = True
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "./checkpoints/Qwen3-VL-4B-Instruct",
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "./checkpoints/Qwen3-VL-4B-Instruct",
        trust_remote_code=True)
    
    # Load and encode image to base64
    image_path = Path(args.image)
    if image_path.exists():
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_base64}"
    else:
        print(f"Warning: Image file {args.image} not found. Using None.")
        image_url = None
    
    messages = [
        {"role": "user", 
         "content": [
             {"type": "image", "image": image_url},
             {"type": "text", "text": """Describe the image."""}
         ]}
    ]
    
    if args.complete:
        print("=== Text Completion Mode ===")
        completion = complete_text(
            model,
            processor,
            messages=messages,
            max_new_tokens=100,
            temperature=1.0
        )
        print(f"Completion:\n{completion}")
    
    if args.next:
        print("=== Next Token Prediction Mode ===")
        predicted_token, logits, probs = predict_next(
            model,
            processor,
            messages=messages,
            constrain_tokens=None,
            top_k=100
        )
        print(f"Predicted Token: '{predicted_token}'")
        print(f"Top probabilities: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:100]}")