import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Tuple
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.utils.action import ACTION_BASE_EMBEDDING, ACTION_TOKENS
except ImportError:
    print("Error: Could not import ACTION_BASE_EMBEDDING from src.utils.action.")
    print("Please ensure you are running this script from the root of the project or PYTHONPATH is set.")
    exit(1)

def load_model_embeddings(model_path: str, device="cuda"):
    """
    Load a model and return its input embedding layer and tokenizer.
    Designed to be memory efficient (load, extract, delete).
    """
    print(f"Loading model components from {model_path}...")
    try:
        # We only need the embedding layer for this visualization, but loading the full model 
        # is usually safer to ensure weight tying is handled if applicable.
        # Using CPU offloading if possible to save VRAM for SFT/Base comparison
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map=device, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
        # Extract Embedding Matrix: shape [vocab_size, hidden_dim]
        embedding_layer = model.get_input_embeddings()
        embeddings = embedding_layer.weight.detach()
        
        return embeddings, tokenizer
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

def get_token_vector(tokenizer, embeddings, token_str: str) -> torch.Tensor:
    """
    Retrieve the embedding vector for a specific token string.
    If the string tokenizes to multiple IDs, we take the mean (common for words).
    """
    # Check if it's a special token first
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    
    # If unk (and not actually unk), it might be a multi-token word like "cursor"
    if token_id == tokenizer.unk_token_id and token_str != tokenizer.unk_token:
        ids = tokenizer(token_str, add_special_tokens=False)['input_ids']
        if not ids:
            print(f"Warning: Token '{token_str}' could not be tokenized.")
            return torch.zeros(embeddings.shape[1], device=embeddings.device)
        # Average the vectors for the sub-words
        vecs = embeddings[ids]
        return vecs.mean(dim=0)
    
    return embeddings[token_id]

def compute_centroid(tokenizer, embeddings, word_list: List[str]) -> torch.Tensor:
    """
    Compute the centroid (average vector) of a list of anchor words.
    E.g., average(["move", "cursor", "up"])
    """
    vecs = []
    for w in word_list:
        v = get_token_vector(tokenizer, embeddings, w)
        vecs.append(v)
    
    if not vecs:
        return torch.zeros(embeddings.shape[1], device=embeddings.device)
    
    return torch.stack(vecs).mean(dim=0)

def plot_semantic_trajectory(
    base_embs, base_tokenizer, 
    sft_embs, sft_tokenizer, 
    token_list, 
    title="Action Token Trajectory (Base -> SFT)"
):
    """
    Visualizes how Action Tokens moved in vector space from Base to SFT.
    Uses PCA to reduce dimensionality.
    """
    print("Generating Trajectory Plot...")
    
    base_vectors = []
    sft_vectors = []
    valid_tokens = []

    for token in token_list:
        v_base = get_token_vector(base_tokenizer, base_embs, token)
        v_sft = get_token_vector(sft_tokenizer, sft_embs, token)
        
        base_vectors.append(v_base.float().cpu().numpy())
        sft_vectors.append(v_sft.float().cpu().numpy())
        valid_tokens.append(token)

    base_vectors = np.array(base_vectors)
    sft_vectors = np.array(sft_vectors)
    
    # Combine for PCA to share the same latent space
    combined = np.concatenate([base_vectors, sft_vectors], axis=0)
    
    pca = PCA(n_components=2)
    result = pca.fit_transform(combined)
    
    n = len(valid_tokens)
    base_2d = result[:n]
    sft_2d = result[n:]
    
    plt.figure(figsize=(14, 12))
    
    for i in range(n):
        plt.arrow(
            base_2d[i, 0], base_2d[i, 1], 
            sft_2d[i, 0] - base_2d[i, 0], sft_2d[i, 1] - base_2d[i, 1],
            color='gray', alpha=0.3, head_width=0.0, length_includes_head=True
        )
        
    # 2. Scatter points
    plt.scatter(base_2d[:, 0], base_2d[:, 1], c='red', alpha=0.5, label='Base Init (Start)', s=80)
    plt.scatter(sft_2d[:, 0], sft_2d[:, 1], c='blue', alpha=1.0, label='SFT Learned (End)', s=100)
    
    from adjustText import adjust_text
    texts = []
    
    for i, txt in enumerate(valid_tokens):
        t1 = plt.text(base_2d[i, 0], base_2d[i, 1], txt, 
                 fontsize=8, color='darkred', alpha=0.7)
        texts.append(t1)

        t2 = plt.text(sft_2d[i, 0], sft_2d[i, 1], txt, 
                 fontsize=10, color='darkblue', weight='bold')
        texts.append(t2)

    try:
        from adjustText import adjust_text
        print("Optimizing text placement (adjustText)...")
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    except ImportError:
        print("Tips: Install 'adjustText' (pip install adjustText) for better label placement.")

    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/viz_trajectory.png", dpi=300)
    print("Saved viz_trajectory.png")

def plot_anchor_alignment(embeddings, tokenizer, action_map, title="SFT Action Tokens vs Semantic Anchors"):
    """
    Plots SFT Action Tokens alongside their Semantic Anchor words.
    Helps visualize if <MOVE_UP> is actually near "Up".
    """
    print("Generating Anchor Alignment Plot...")
    
    target_vectors = []
    labels = []
    colors = []
    markers = []
    
    # Define a color palette for groups
    palette = sns.color_palette("hsv", len(action_map))
    
    for i, (action_token, anchors) in enumerate(action_map.items()):
        # 1. The Action Token itself
        v_action = get_token_vector(tokenizer, embeddings, action_token)
        target_vectors.append(v_action.float().cpu().numpy())
        labels.append(action_token)
        colors.append(palette[i])
        markers.append('o') # Circle for Action Token
        
        # 2. The Anchors (Average centroid to reduce noise)
        v_anchor = compute_centroid(tokenizer, embeddings, anchors)
        target_vectors.append(v_anchor.float().cpu().numpy())
        labels.append(f"Anchor({action_token})")
        colors.append(palette[i])
        markers.append('x') # X for Natural Language Anchor
        
    target_vectors = np.array(target_vectors)
    
    # Use t-SNE here as we have distinct clusters
    # Perplexity must be < n_samples.
    perp = min(30, len(target_vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    result = tsne.fit_transform(target_vectors)
    
    plt.figure(figsize=(14, 12))
    
    for i in range(len(target_vectors)):
        plt.scatter(
            result[i, 0], result[i, 1], 
            color=colors[i], 
            marker=markers[i],
            s=100 if markers[i] == 'o' else 50
        )
        
        # Draw line between token and its anchor
        if i % 2 == 1: # This is an anchor, connect to previous (action)
            plt.plot(
                [result[i-1, 0], result[i, 0]],
                [result[i-1, 1], result[i, 1]],
                color=colors[i], alpha=0.2, linestyle='--'
            )
            
            # Simple label
            clean_label = labels[i-1] # The action token name
            plt.text(result[i-1, 0], result[i-1, 1]+0.2, clean_label, fontsize=8)

    plt.title(title)
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
                    Line2D([0], [0], marker='x', color='gray', markersize=10)]
    plt.legend(custom_lines, ['Action Token', 'NL Anchor Centroid'])
    
    plt.tight_layout()
    plt.savefig("./results/viz_anchors.png", dpi=300)
    print("Saved viz_anchors.png")

def plot_similarity_heatmap(embeddings, tokenizer, token_list):
    """
    Plots a Cosine Similarity Heatmap between all Action Tokens.
    Useful to check orthogonality (e.g. is UP correlated with DOWN?).
    """
    print("Generating Heatmap...")
    vectors = []
    for t in token_list:
        vectors.append(get_token_vector(tokenizer, embeddings, t))
    
    vectors = torch.stack(vectors).float()
    
    # Normalize
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    
    # Compute Cosine Similarity Matrix
    sim_matrix = torch.mm(vectors, vectors.t()).cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_matrix, 
        xticklabels=token_list, 
        yticklabels=token_list, 
        cmap="coolwarm", 
        center=0,
        annot=False # Turn on if list is small
    )
    plt.title("Cosine Similarity between Action Tokens")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("./results/viz_heatmap.png", dpi=300)
    print("Saved viz_heatmap.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", type=str, required=True, help="Path to SFT model checkpoint")
    parser.add_argument("--base_model", type=str, default=None, help="Path to Base model (optional, for trajectory)")
    args = parser.parse_args()

    # 1. Load SFT Model
    sft_embs, sft_tokenizer = load_model_embeddings(args.sft_model)
    if sft_embs is None:
        exit(1)

    # 2. Visualization: Anchors & Heatmap (Only needs SFT)
    # Filter tokens to ensure they exist in the map
    tokens_to_plot = [t for t in ACTION_TOKENS if t in ACTION_BASE_EMBEDDING]
    
    plot_anchor_alignment(sft_embs, sft_tokenizer, ACTION_BASE_EMBEDDING)
    plot_similarity_heatmap(sft_embs, sft_tokenizer, tokens_to_plot)

    # 3. Visualization: Trajectory (Needs Base)
    if args.base_model:
        # Clear VRAM to make room for Base model if needed
        # In a real script, we might want to keep sft_embs on CPU
        del sft_embs
        torch.cuda.empty_cache()
        
        # Reload SFT embeddings to CPU to compare
        sft_embs_cpu, _ = load_model_embeddings(args.sft_model, device="cpu")
        
        # Load Base
        base_embs, base_tokenizer = load_model_embeddings(args.base_model)
        
        if base_embs is not None:
            plot_semantic_trajectory(
                base_embs, base_tokenizer, 
                sft_embs_cpu, sft_tokenizer, 
                tokens_to_plot
            )
        else:
            print("Skipping trajectory plot (Base model load failed).")
    else:
        print("Skipping trajectory plot (No base model provided).")
    
    print("Done.")