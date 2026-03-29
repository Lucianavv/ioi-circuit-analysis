import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict, Optional


def run_with_cache(
    model: HookedTransformer,
    prompt_text: str
) -> Tuple[torch.Tensor, object]:
    """
    Returns:
        Tuple of (logits, cache).
        logits: shape [1, seq_len, vocab_size]
        cache: ActivationCache object.
    """
    tokens = model.to_tokens(prompt_text)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache


# Attention pattern extraction
def get_attention_pattern(
    cache,
    layer: int,
    head: int
) -> torch.Tensor:
    """
    Returns:
        Attention pattern tensor of shape [seq_len, seq_len].
        Entry [i, j] is the attention weight from position i to position j.
    """
    return cache["pattern", layer][0, head, :, :]


def get_name_attention_ratio(
    model: HookedTransformer,
    cache,
    tokens: torch.Tensor,
    layer: int,
    head: int,
    correct_name: str,
    incorrect_name: str
) -> float:
    """
    Attention ratio = sum(attn to correct name positions) /
                      sum(attn to incorrect name positions)

    Returns:
        Attention ratio as a float.
    """
    head_pattern = get_attention_pattern(cache, layer, head)
    final_pos = tokens.shape[1] - 1
    attention_from_final = head_pattern[final_pos, :]

    token_strs = [model.to_string(tokens[0, j]) for j in range(tokens.shape[1])]

    correct_positions = [j for j, t in enumerate(token_strs) if t == " " + correct_name]
    incorrect_positions = [j for j, t in enumerate(token_strs) if t == " " + incorrect_name]

    attn_correct = sum(attention_from_final[p].item() for p in correct_positions)
    attn_incorrect = sum(attention_from_final[p].item() for p in incorrect_positions)

    return attn_correct / (attn_incorrect + 1e-10)

# Multi-prompt validation
def validate_heads_across_prompts(
    model: HookedTransformer,
    dataset: list,
    heads: List[Tuple[int, int]],
    n_samples: int = 10
) -> Dict[str, List[float]]:

    head_ratios: Dict[str, List[float]] = {f"{l}.{h}": [] for l, h in heads}

    for i in range(min(n_samples, len(dataset))):
        prompt_data = dataset[i]
        tokens = model.to_tokens(prompt_data["prompt"])
        _, cache = model.run_with_cache(tokens)

        for layer, head in heads:
            ratio = get_name_attention_ratio(
                model, cache, tokens, layer, head,
                prompt_data["correct_answer"],
                prompt_data["incorrect_answer"]
            )
            head_ratios[f"{layer}.{head}"].append(ratio)

    return head_ratios


def summarize_head_ratios(head_ratios: Dict[str, List[float]]) -> None:
    print(f"{'Head':<10} {'Mean Ratio':<15} {'Std':<10} {'Assessment'}")
    print("=" * 55)
    for head_name, ratios in head_ratios.items():
        mean = np.mean(ratios)
        std = np.std(ratios)
        assessment = "Name mover" if mean > 2.0 else "Weak / non-mover"
        print(f"{head_name:<10} {mean:<15.2f} {std:<10.2f} {assessment}")
