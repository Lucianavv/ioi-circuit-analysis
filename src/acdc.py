import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Tuple, Set, Dict
from src.metrics import compute_logit_difference
from src.ioi_dataset import create_corrupted_prompt


class Edge:
    """
    Represents a directed connection between two attention heads.
    """
    def __init__(self, sender_layer, sender_head,
                 receiver_layer, receiver_head):
        self.sender_layer = sender_layer
        self.sender_head = sender_head
        self.receiver_layer = receiver_layer
        self.receiver_head = receiver_head

    def __repr__(self):
        return (f"Edge({self.sender_layer}.{self.sender_head} -> "
                f"{self.receiver_layer}.{self.receiver_head})")

    def __hash__(self):
        return hash((self.sender_layer, self.sender_head,
                     self.receiver_layer, self.receiver_head))

    def __eq__(self, other):
        return (self.sender_layer == other.sender_layer and
                self.sender_head == other.sender_head and
                self.receiver_layer == other.receiver_layer and
                self.receiver_head == other.receiver_head)


def compute_baseline(
    model: HookedTransformer,
    dataset: list,
    ablated_heads: Set[Tuple[int, int]],
    n_samples: int = 50
) -> float:
    """
    Compute mean logit difference with ablated_heads zeroed out.
    """
    logit_diffs = []

    for i in range(min(n_samples, len(dataset))):
        prompt_data = dataset[i]
        tokens = model.to_tokens(prompt_data['prompt'])

        if len(ablated_heads) == 0:
            logits = model(tokens)
        else:
            hooks = []
            for (layer, head) in ablated_heads:
                def make_hook(h):
                    def hook_fn(z, hook):
                        z[:, :, h, :] = 0.0
                        return z
                    return hook_fn
                hooks.append(
                    (f"blocks.{layer}.attn.hook_z", make_hook(head)))
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        final_logits = logits[0, -1, :]
        correct_id = model.to_single_token(
            " " + prompt_data['correct_answer'])
        incorrect_id = model.to_single_token(
            " " + prompt_data['incorrect_answer'])
        diff = (final_logits[correct_id] -
                final_logits[incorrect_id]).item()
        logit_diffs.append(diff)

    return float(np.mean(logit_diffs))


def score_edge(
    model: HookedTransformer,
    dataset: list,
    sender_layer: int,
    sender_head: int,
    ablated_heads: Set[Tuple[int, int]],
    n_samples: int = 20
) -> float:
    """
    Score a head by patching it with corrupted activations and
    measuring the drop in logit difference.

    Drop = clean_logit_diff - patched_logit_diff
    """
    drops = []

    for i in range(min(n_samples, len(dataset))):
        clean = dataset[i]
        corrupted = create_corrupted_prompt(clean)

        clean_tokens = model.to_tokens(clean['prompt'])
        corrupted_tokens = model.to_tokens(corrupted['prompt'])

        # Cache corrupted activation for this head
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)
        corrupted_act = corrupted_cache[
            f"blocks.{sender_layer}.attn.hook_z"
        ][0, :, sender_head, :].clone()

        # Build hooks: patch this head + zero ablated heads
        hooks = []

        def make_patch_hook(act, h):
            def hook_fn(z, hook):
                z[:, :, h, :] = act
                return z
            return hook_fn

        hooks.append((
            f"blocks.{sender_layer}.attn.hook_z",
            make_patch_hook(corrupted_act, sender_head)
        ))

        for (layer, head) in ablated_heads:
            if layer == sender_layer and head == sender_head:
                continue
            def make_zero_hook(h):
                def hook_fn(z, hook):
                    z[:, :, h, :] = 0.0
                    return z
                return hook_fn
            hooks.append((
                f"blocks.{layer}.attn.hook_z",
                make_zero_hook(head)
            ))

        # Patched run
        patched_logits = model.run_with_hooks(
            clean_tokens, fwd_hooks=hooks)
        patched_final = patched_logits[0, -1, :]
        correct_id = model.to_single_token(
            " " + clean['correct_answer'])
        incorrect_id = model.to_single_token(
            " " + clean['incorrect_answer'])
        patched_diff = (patched_final[correct_id] -
                        patched_final[incorrect_id]).item()

        # Clean run (with ablated heads zeroed)
        if len(ablated_heads) == 0:
            clean_logits = model(clean_tokens)
        else:
            clean_hooks = []
            for (layer, head) in ablated_heads:
                def make_zh(h):
                    def hf(z, hook):
                        z[:, :, h, :] = 0.0
                        return z
                    return hf
                clean_hooks.append((
                    f"blocks.{layer}.attn.hook_z",
                    make_zh(head)
                ))
            clean_logits = model.run_with_hooks(
                clean_tokens, fwd_hooks=clean_hooks)

        clean_final = clean_logits[0, -1, :]
        clean_diff = (clean_final[correct_id] -
                      clean_final[incorrect_id]).item()

        drops.append(clean_diff - patched_diff)

    return float(np.mean(drops))



# Main ACDC algorithm

def run_acdc(
    model: HookedTransformer,
    dataset: list,
    threshold: float = 0.1,
    n_samples: int = 50,
    score_samples: int = 20,
    verbose: bool = True
) -> Tuple[Set[Tuple[int, int]], Dict]:
    """

    Returns:
        circuit_heads: Set of (layer, head) tuples in discovered circuit.
        metrics: Dict with faithfulness, sparsity, forward_passes.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    forward_pass_count = 0

    all_heads = {(l, h) for l in range(n_layers)
                 for h in range(n_heads)}
    ablated_heads: Set[Tuple[int, int]] = set()

    # Initial baseline
    initial_baseline = compute_baseline(
        model, dataset, ablated_heads, n_samples)
    forward_pass_count += n_samples

    if verbose:
        print(f"Initial baseline (full model): {initial_baseline:.3f}")
        print(f"Threshold tau: {threshold}")
        print(f"Evaluating {n_layers * n_heads} heads...\n")

    # Traverse layers backwards
    for layer in range(n_layers - 1, -1, -1):
        for head in range(n_heads):

            drop = score_edge(
                model, dataset, layer, head,
                ablated_heads, score_samples)
            forward_pass_count += score_samples * 2

            if drop < threshold:
                ablated_heads.add((layer, head))
                forward_pass_count += n_samples  # baseline recompute

                if verbose:
                    print(f"  PRUNED {layer}.{head:>2} "
                          f"(drop={drop:.4f} < {threshold})")
            else:
                if verbose:
                    print(f"  KEPT   {layer}.{head:>2} "
                          f"(drop={drop:.4f} >= {threshold})")

    # Circuit = all heads not ablated
    circuit_heads = all_heads - ablated_heads

    # Final metrics
    final_baseline = compute_baseline(
        model, dataset, ablated_heads, n_samples)
    forward_pass_count += n_samples

    faithfulness = (final_baseline / initial_baseline
                    if initial_baseline != 0 else 0.0)
    sparsity = 1.0 - (len(circuit_heads) / (n_layers * n_heads))

    metrics = {
        "faithfulness": faithfulness,
        "sparsity": sparsity,
        "forward_passes": forward_pass_count,
        "circuit_size": len(circuit_heads),
        "initial_baseline": initial_baseline,
        "final_baseline": final_baseline,
        "threshold": threshold,
        "n_pruned": len(ablated_heads),
        "n_kept": len(circuit_heads),
        "circuit_heads": sorted(list(circuit_heads))
    }

    if verbose:
        print(f"\nACDC complete.")
        print(f"  Circuit heads: {sorted(circuit_heads)}")
        print(f"  Circuit size:  {len(circuit_heads)}")
        print(f"  Faithfulness:  {faithfulness:.3f}")
        print(f"  Sparsity:      {sparsity:.3f}")
        print(f"  Forward passes: {forward_pass_count}")

    return circuit_heads, metrics



# Utilities

def get_circuit_heads(circuit):
    return circuit


def compute_circuit_overlap(circuit_a, circuit_b):
    if not circuit_a and not circuit_b:
        return 1.0
    intersection = len(circuit_a & circuit_b)
    union = len(circuit_a | circuit_b)
    return intersection / union if union > 0 else 0.0
