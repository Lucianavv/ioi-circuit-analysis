import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Tuple, List, Optional


def compute_logit_difference(
    model: HookedTransformer,
    prompt_text: str,
    correct_name: str,
    incorrect_name: str
) -> Tuple[float, float, float]:
    """
    Compute logit difference for a single IOI prompt.

    Logit difference = Logit(correct_name) - Logit(incorrect_name)
    measured at the final token position.

    Returns:
        Tuple of (logit_diff, correct_logit, incorrect_logit).
    """
    tokens = model.to_tokens(prompt_text)
    logits = model(tokens)
    final_logits = logits[0, -1, :]

    correct_token_id = model.to_single_token(" " + correct_name)
    incorrect_token_id = model.to_single_token(" " + incorrect_name)

    correct_logit = final_logits[correct_token_id].item()
    incorrect_logit = final_logits[incorrect_token_id].item()

    return correct_logit - incorrect_logit, correct_logit, incorrect_logit


def compute_logit_difference_from_logits(
    logits: torch.Tensor,
    model: HookedTransformer,
    correct_name: str,
    incorrect_name: str
) -> float:
    """
    Compute logit difference from a pre-computed logits tensor.

    Args:
        logits: Shape [batch, seq_len, vocab_size] or [vocab_size].
        model: Used only for token lookup.
        correct_name: The correct answer name.
        incorrect_name: The incorrect answer name.

    Returns:
        Scalar logit difference.
    """
    if logits.dim() == 3:
        final_logits = logits[0, -1, :]
    else:
        final_logits = logits

    correct_token_id = model.to_single_token(" " + correct_name)
    incorrect_token_id = model.to_single_token(" " + incorrect_name)

    return (final_logits[correct_token_id] - final_logits[incorrect_token_id]).item()
    

def evaluate_dataset(
    model: HookedTransformer,
    dataset: list,
    n_samples: Optional[int] = None
) -> dict:

    #Compute baseline logit difference statistics over a dataset.

    if n_samples is not None:
        dataset = dataset[:n_samples]

    logit_diffs = []

    for prompt_data in dataset:
        diff, _, _ = compute_logit_difference(
            model,
            prompt_data["prompt"],
            prompt_data["correct_answer"],
            prompt_data["incorrect_answer"]
        )
        logit_diffs.append(diff)

    logit_diffs = np.array(logit_diffs)

    # Per-template breakdown
    by_template = {}
    for template in ["template_1", "template_2", "template_3"]:
        indices = [i for i, p in enumerate(dataset) if p["template"] == template]
        if indices:
            subset = logit_diffs[indices]
            by_template[template] = {
                "mean": float(subset.mean()),
                "accuracy": float((subset > 0).sum() / len(subset))
            }

    return {
        "logit_diffs": logit_diffs,
        "mean": float(logit_diffs.mean()),
        "median": float(np.median(logit_diffs)),
        "std": float(logit_diffs.std()),
        "min": float(logit_diffs.min()),
        "max": float(logit_diffs.max()),
        "accuracy": float((logit_diffs > 0).sum() / len(logit_diffs)),
        "n_samples": len(logit_diffs),
        "by_template": by_template
    }


def compute_faithfulness(
    circuit_logit_diffs: np.ndarray,
    full_model_logit_diffs: np.ndarray
) -> float:
    """
    Compute circuit faithfulness as the ratio of mean logit differences.

    Faithfulness = mean(circuit_LD) / mean(full_model_LD)

    """
    return float(circuit_logit_diffs.mean() / full_model_logit_diffs.mean())
