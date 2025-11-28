import torch
import argparse
import base64
import io
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor,
)
from .prompts import AGENT_SYSTEM_PROMPT

# Function to predict the next token probabilities
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
    """
    tokenizer = processor.tokenizer
    
    # Prepare input text using chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Process inputs (tokenize and handle images)
    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the next token (last position)
        next_token_logits = outputs.logits[0, -1, :] 
    
    # Apply temperature scaling
    scaled_logits = next_token_logits / temperature
    
    # Apply token constraints if specified
    if constrain_tokens is not None:
        constrain_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in constrain_tokens]
        constrain_token_ids_tensor = torch.tensor(constrain_token_ids, device=model.device)
        
        # Mask irrelevant tokens
        mask = torch.full_like(scaled_logits, float('-inf'))
        mask[constrain_token_ids_tensor] = 0
        scaled_logits = scaled_logits + mask
        
        constrained_logits = next_token_logits[constrain_token_ids_tensor]
        tokens_to_report = constrain_tokens
        token_ids_to_report = constrain_token_ids_tensor
    else:
        constrained_logits = scaled_logits
        
        # Apply top_k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(scaled_logits, k=min(top_k, scaled_logits.size(-1)))
            
            mask = torch.full_like(scaled_logits, float('-inf'))
            mask[top_k_indices] = 0
            scaled_logits = scaled_logits + mask
            
            constrained_logits = next_token_logits[top_k_indices]
            tokens_to_report = [tokenizer.decode([idx]) for idx in top_k_indices.cpu().tolist()]
            token_ids_to_report = top_k_indices
        else:
            # Report top 50 if no k specified
            top_50_logits, top_50_indices = torch.topk(next_token_logits, k=50)
            constrained_logits = top_50_logits
            tokens_to_report = [tokenizer.decode([idx]) for idx in top_50_indices.cpu().tolist()]
            token_ids_to_report = top_50_indices
    
    # Calculate probabilities
    probabilities = torch.softmax(scaled_logits, dim=0)
    
    # Sample predicted token
    sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
    predicted_token = tokenizer.decode([sampled_idx])
    
    reported_probs = probabilities[token_ids_to_report]
    
    # Build result dictionaries
    logits_dict = {token: logits.item() 
                   for token, logits in zip(tokens_to_report, constrained_logits.cpu())}
    probs_dict = {token: prob.item() 
                  for token, prob in zip(tokens_to_report, reported_probs.cpu())}
    
    return predicted_token, logits_dict, probs_dict


# Function to generate full text completion
def complete_text(
    model,
    processor,
    messages,
    max_new_tokens=100,
    temperature=1.0,
):
    """
    Generate a completion for the given messages.
    """
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    completion = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return completion


def resize_image(image, target_w=None, target_h=None):
    """
    Resize image based on target dimensions.
    Handles aspect ratio if only one dimension is provided.
    """
    if target_w is None and target_h is None:
        return image
    
    orig_w, orig_h = image.size
    
    # Calculate missing dimension to maintain aspect ratio
    if target_w is not None and target_h is None:
        ratio = target_w / orig_w
        target_h = int(orig_h * ratio)
    elif target_h is not None and target_w is None:
        ratio = target_h / orig_h
        target_w = int(orig_w * ratio)
        
    print(f"Resizing image from ({orig_w}, {orig_h}) to ({target_w}, {target_h})")
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test model prediction")
    parser.add_argument("-c", "--complete", action="store_true", 
                        help="Complete the text instead of predicting next token")
    parser.add_argument("-n", "--next", action="store_true",
                        help="Predict next token (default behavior)")
    parser.add_argument("-i", "--image", type=str, default="./image.png",
                        help="Path to image file (default: ./image.png)")
    # New arguments for resizing
    parser.add_argument("--width", type=int, default=None,
                        help="Target width for image resize")
    parser.add_argument("--height", type=int, default=None,
                        help="Target height for image resize")
    
    args = parser.parse_args()
    
    # Default to next token mode
    if not args.complete and not args.next:
        args.next = True
    
    # Load model and processor
    model_path = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot" # Update path as needed
    print(f"Loading model from {model_path}...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Process image
    image_path = Path(args.image)
    image_url = None
    
    if image_path.exists():
        try:
            # Open image using PIL
            with Image.open(image_path) as img:
                # Resize if arguments provided
                if args.width or args.height:
                    img = resize_image(img, args.width, args.height)
                
                # Convert to base64
                buffered = io.BytesIO()
                # Save as PNG to buffer
                img_format = img.format if img.format else 'PNG'
                img.save(buffered, format=img_format)
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                mime_type = f"image/{img_format.lower()}"
                image_url = f"data:{mime_type};base64,{image_base64}"
                
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print(f"Warning: Image file {args.image} not found. Using text-only input.")

    # Construct messages
    content = []
    #if image_url:
        #content.append({"type": "image", "image": image_url})
    content.append({"type": "text", "text": "Dont't go down. Go the opposite way."})

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    
    # Execution modes
    if args.complete:
        print("=== Text Completion Mode ===")
        completion = complete_text(
            model,
            processor,
            messages=messages,
            max_new_tokens=512,
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
        print(f"Top probabilities: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:20]}")