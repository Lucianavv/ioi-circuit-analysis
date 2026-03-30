import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, Set, Tuple
from src.ioi_dataset import create_corrupted_prompt


def compute_attribution_scores_single(
    model: HookedTransformer,
    clean_prompt: dict,
    corrupted_prompt: dict
) -> Dict[Tuple[int, int], float]:

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Forward pass 1: cache clean activations 
    clean_tokens = model.to_tokens(clean_prompt['prompt'])
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)

    # Forward pass 2: cache corrupted activations 
    corrupted_tokens = model.to_tokens(corrupted_prompt['prompt'])
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)


    model.zero_grad()

    hook_z_tensors = {}

    def make_save_hook(layer):
        def hook_fn(z, hook):
            z = z.clone().detach().requires_grad_(True)
            hook_z_tensors[layer] = z
            return z
        return hook_fn

    hooks = [(f"blocks.{l}.attn.hook_z", make_save_hook(l))
             for l in range(n_layers)]

    logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

    # Compute logit difference
    final_logits = logits[0, -1, :]
    correct_id = model.to_single_token(
        " " + clean_prompt['correct_answer'])
    incorrect_id = model.to_single_token(
        " " + clean_prompt['incorrect_answer'])
    logit_diff = final_logits[correct_id] - final_logits[incorrect_id]

    # Backward pass
    logit_diff.backward()

    scores = {}
    for layer in range(n_layers):
        if layer not in hook_z_tensors:
            continue

        z_clean = hook_z_tensors[layer]  # [1, seq, n_heads, d_head]
        grad = z_clean.grad              # same shape

        if grad is None:
            for head in range(n_heads):
                scores[(layer, head)] = 0.0
            continue

        z_corrupted = corrupted_cache[
            f"blocks.{layer}.attn.hook_z"]  # [1, seq, n_heads, d_head]

        # activation difference
        act_diff = (z_clean - z_corrupted).detach()

        # element-wise product then sum over d_head,
        # average over seq and batch
        # result shape per head: scalar
        for head in range(n_heads):
            score = (grad[:, :, head, :] *
                     act_diff[:, :, head, :]).sum(dim=-1).mean().item()
            scores[(layer, head)] = score

    return scores

# Main Attribution Patching algorithm

def run_attribution_patching(
    model: HookedTransformer,
    dataset: list,
    threshold: float = 0.05,
    n_samples: int = 50,
    verbose: bool = True
) -> Tuple[Set[Tuple[int, int]], Dict]:
    """
    Returns:
        circuit_heads: Set of (layer, head) tuples in circuit.
        metrics: Dict with faithfulness, sparsity, pass counts.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    forward_pass_count = 0
    backward_pass_count = 0

    if verbose:
        print(f"Computing attribution scores over {n_samples} prompts...")

    # Accumulate scores across prompts
    accumulated_scores: Dict[Tuple[int, int], float] = {
        (l, h): 0.0
        for l in range(n_layers)
        for h in range(n_heads)
    }

    for i in range(min(n_samples, len(dataset))):
        clean = dataset[i]
        corrupted = create_corrupted_prompt(clean)

        scores = compute_attribution_scores_single(
            model, clean, corrupted)

        for key, val in scores.items():
            accumulated_scores[key] += val

        forward_pass_count += 2
        backward_pass_count += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples} prompts...")

    # Average scores
    mean_scores = {
        key: val / n_samples
        for key, val in accumulated_scores.items()
    }

    # Build circuit by thresholding absolute scores
    circuit_heads = {
        (l, h)
        for (l, h), score in mean_scores.items()
        if abs(score) >= threshold
    }

    if verbose:
        print(f"\nAttribution scores computed.")
        print(f"Threshold: {threshold}")
        print(f"Circuit size: {len(circuit_heads)} heads")

    # Ablate all heads NOT in circuit, measure logit difference recovery
    ablated_heads = {
        (l, h) for l in range(n_layers)
        for h in range(n_heads)
        if (l, h) not in circuit_heads
    }

    # Full model baseline
    full_diffs = []
    circuit_diffs = []

    for i in range(min(n_samples, len(dataset))):
        prompt = dataset[i]
        tokens = model.to_tokens(prompt['prompt'])
        correct_id = model.to_single_token(
            " " + prompt['correct_answer'])
        incorrect_id = model.to_single_token(
            " " + prompt['incorrect_answer'])

        # Full model
        full_logits = model(tokens)
        full_diff = (full_logits[0, -1, correct_id] -
                     full_logits[0, -1, incorrect_id]).item()
        full_diffs.append(full_diff)
        forward_pass_count += 1

        # Circuit only (ablate non-circuit heads)
        if len(ablated_heads) > 0:
            hooks = []
            for (layer, head) in ablated_heads:
                def make_hook(h):
                    def hf(z, hook):
                        z[:, :, h, :] = 0.0
                        return z
                    return hf
                hooks.append(
                    (f"blocks.{layer}.attn.hook_z", make_hook(head)))
            circuit_logits = model.run_with_hooks(
                tokens, fwd_hooks=hooks)
        else:
            circuit_logits = model(tokens)
        forward_pass_count += 1

        circuit_diff = (circuit_logits[0, -1, correct_id] -
                        circuit_logits[0, -1, incorrect_id]).item()
        circuit_diffs.append(circuit_diff)

    full_mean = float(np.mean(full_diffs))
    circuit_mean = float(np.mean(circuit_diffs))
    faithfulness = circuit_mean / full_mean if full_mean != 0 else 0.0
    sparsity = 1.0 - (len(circuit_heads) / (n_layers * n_heads))

    metrics = {
        "faithfulness": faithfulness,
        "sparsity": sparsity,
        "forward_passes": forward_pass_count,
        "backward_passes": backward_pass_count,
        "circuit_size": len(circuit_heads),
        "full_model_mean_ld": full_mean,
        "circuit_mean_ld": circuit_mean,
        "threshold": threshold,
        "mean_scores": mean_scores,
        "circuit_heads": sorted(list(circuit_heads))
    }

    if verbose:
        print(f"\nAttribution Patching complete.")
        print(f"  Circuit heads: {sorted(circuit_heads)}")
        print(f"  Circuit size:  {len(circuit_heads)}")
        print(f"  Faithfulness:  {faithfulness:.3f}")
        print(f"  Sparsity:      {sparsity:.3f}")
        print(f"  Forward passes: {forward_pass_count}")
        print(f"  Backward passes: {backward_pass_count}")

    return circuit_heads, metrics


# Visualization utility

def scores_to_grid(
    scores: Dict[Tuple[int, int], float],
    n_layers: int,
    n_heads: int
) -> np.ndarray:
    """
    Convert attribution scores to [n_layers, n_heads] grid
    for heatmap visualization.
    """
    grid = np.zeros((n_layers, n_heads))
    for (layer, head), score in scores.items():
        grid[layer, head] = score
    return grid
