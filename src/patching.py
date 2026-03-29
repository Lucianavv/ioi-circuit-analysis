"""
patching.py
-----------
Activation patching infrastructure for IOI circuit analysis.

Implements three patching strategies:
  1. Zero ablation   — replace a head's output with zeros.
  2. Mean ablation   — replace with the mean activation across a reference dataset.
  3. Path patching   — substitute corrupted activations into a clean forward pass.

Path patching is the primary tool for causal circuit discovery: by patching
corrupted activations into specific components of a clean run, we measure
how much each component contributes to the model's IOI performance.

Convention (Wang et al. 2022): patch corrupted → clean.
  - Start with clean tokens.
  - Inject corrupted activations at the target component.
  - Measure drop in logit difference on clean labels.
  A large drop = the component carries task-relevant information.

Used in: Deliverable 1 (validation), Deliverable 2 (ACDC & Attribution Patching).
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mean activation computation
# ---------------------------------------------------------------------------

def get_mean_head_activation(
    model: HookedTransformer,
    dataset: list,
    layer: int,
    head: int,
    n_samples: int = 50
) -> torch.Tensor:
    """
    Compute the mean output activation of an attention head across a dataset.

    Averages over both prompts and sequence positions to produce a single
    d_head-dimensional reference vector used for mean ablation.

    Args:
        model: HookedTransformer instance.
        dataset: List of prompt dicts.
        layer: Layer index.
        head: Head index.
        n_samples: Number of prompts to average over.

    Returns:
        Mean activation tensor of shape [d_head].
    """
    mean_activation = None
    n = min(n_samples, len(dataset))

    for i in range(n):
        tokens = model.to_tokens(dataset[i]["prompt"])
        _, cache = model.run_with_cache(tokens)

        # z: [batch, pos, n_heads, d_head]
        z = cache[f"blocks.{layer}.attn.hook_z"]
        head_act = z[0, :, head, :]  # [pos, d_head]

        contribution = head_act.mean(dim=0)
        mean_activation = contribution if mean_activation is None else mean_activation + contribution

    return mean_activation / n


# ---------------------------------------------------------------------------
# Ablation methods
# ---------------------------------------------------------------------------

def zero_ablate_head(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    head: int
) -> torch.Tensor:
    """
    Run the model with a head's output replaced by zeros.

    Zero ablation is a baseline intervention. It's cruder than mean ablation
    because it injects an out-of-distribution signal, but useful for fast
    importance screening.

    Args:
        model: HookedTransformer instance.
        tokens: Token tensor shape [1, seq_len].
        layer: Layer to ablate.
        head: Head to ablate.

    Returns:
        Logits tensor shape [1, seq_len, vocab_size].
    """
    def hook(z, hook):
        z[:, :, head, :] = 0.0
        return z

    return model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)]
    )


def mean_ablate_head(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    head: int,
    mean_activation: torch.Tensor
) -> torch.Tensor:
    """
    Run the model with a head's output replaced by its mean activation.

    Mean ablation is preferred over zero ablation because the replacement
    value is in-distribution — it represents 'average' rather than 'absent'
    behavior, producing cleaner causal estimates.

    Args:
        model: HookedTransformer instance.
        tokens: Token tensor shape [1, seq_len].
        layer: Layer to ablate.
        head: Head to ablate.
        mean_activation: Reference activation of shape [d_head], from
                         get_mean_head_activation().

    Returns:
        Logits tensor shape [1, seq_len, vocab_size].
    """
    def hook(z, hook):
        # z: [batch, pos, n_heads, d_head]
        batch_size, seq_len = z.shape[0], z.shape[1]
        z[:, :, head, :] = mean_activation.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        return z

    return model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)]
    )


# ---------------------------------------------------------------------------
# Path patching
# ---------------------------------------------------------------------------

def path_patch_head(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer: int,
    head: int
) -> torch.Tensor:
    """
    Patch corrupted activations into a clean forward pass at a single head.

    Procedure:
      1. Run corrupted tokens to obtain corrupted head activations.
      2. Run clean tokens with a hook that replaces this head's output
         with the corrupted activations.

    The resulting logits reflect what happens when only this component
    receives the wrong (corrupted) signal, while everything else runs
    on the clean input.

    Convention: corrupted → clean (Wang et al. 2022).

    Args:
        model: HookedTransformer instance.
        clean_tokens: Token tensor for the clean prompt, shape [1, seq_len].
        corrupted_tokens: Token tensor for the corrupted prompt, shape [1, seq_len].
        layer: Layer to patch.
        head: Head to patch.

    Returns:
        Logits tensor shape [1, seq_len, vocab_size].
    """
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
    corrupted_activation = corrupted_cache[f"blocks.{layer}.attn.hook_z"][0, :, head, :]

    def hook(z, hook):
        z[:, :, head, :] = corrupted_activation
        return z

    return model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)]
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_patching_across_dataset(
    model: HookedTransformer,
    dataset: list,
    heads: List[Tuple[int, int]],
    patch_fn,
    n_samples: int = 50,
    **patch_kwargs
) -> Dict[str, Dict]:
    """
    Evaluate the effect of a patching intervention across multiple prompts.

    For each head and each prompt, computes:
      - clean baseline logit difference
      - patched logit difference
      - drop = clean - patched (positive = head is important)
      - recovery = patched / clean (1.0 = full recovery)

    Args:
        model: HookedTransformer instance.
        dataset: List of prompt dicts (must include corrupted pairs for path patching).
        heads: List of (layer, head) tuples to evaluate.
        patch_fn: One of zero_ablate_head, mean_ablate_head, path_patch_head.
        n_samples: Number of prompts to evaluate.
        **patch_kwargs: Additional arguments passed to patch_fn (e.g. mean_activation).

    Returns:
        Dict mapping "layer.head" → {mean_drop, std_drop, mean_recovery, drops (array)}.
    """
    from src.ioi_dataset import create_corrupted_prompt
    from src.metrics import compute_logit_difference_from_logits

    results = {f"{l}.{h}": {"drops": [], "recoveries": []} for l, h in heads}

    for i in range(min(n_samples, len(dataset))):
        clean = dataset[i]
        corrupted = create_corrupted_prompt(clean)

        clean_tokens = model.to_tokens(clean["prompt"])
        corrupted_tokens = model.to_tokens(corrupted["prompt"])

        clean_diff, _, _ = __import__("src.metrics", fromlist=["compute_logit_difference"]).compute_logit_difference(
            model, clean["prompt"], clean["correct_answer"], clean["incorrect_answer"]
        )

        for layer, head in heads:
            if patch_fn.__name__ == "path_patch_head":
                logits = patch_fn(model, clean_tokens, corrupted_tokens, layer, head)
            else:
                logits = patch_fn(model, clean_tokens, layer, head, **patch_kwargs.get(f"{layer}.{head}", {}))

            patched_diff = compute_logit_difference_from_logits(
                logits, model, clean["correct_answer"], clean["incorrect_answer"]
            )

            drop = clean_diff - patched_diff
            recovery = patched_diff / (clean_diff + 1e-10)

            results[f"{layer}.{head}"]["drops"].append(drop)
            results[f"{layer}.{head}"]["recoveries"].append(recovery)

    # Summarize
    summary = {}
    for key, vals in results.items():
        drops = np.array(vals["drops"])
        recoveries = np.array(vals["recoveries"])
        summary[key] = {
            "mean_drop": float(drops.mean()),
            "std_drop": float(drops.std()),
            "mean_recovery": float(recoveries.mean()),
            "drops": drops
        }

    return summary
