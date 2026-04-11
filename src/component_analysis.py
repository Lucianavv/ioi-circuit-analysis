"""
component_analysis.py

Reusable methods to classify attention heads in discovered circuits
by functional type, following Wang et al. (2022) methodology.

Supports: name movers, negative name movers, S-inhibition,
          induction, duplicate token, previous token heads.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Tuple, List, Dict


#  Attention pattern utilities

def get_mean_attention_pattern(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    token_position: int = -1,
    batch_size: int = 50
) -> np.ndarray:
    """
    Compute mean attention pattern for a head across a dataset.
    
    Returns array of shape [seq_len] — mean attention weights from
    token_position to all other positions.
    """
    all_patterns = []
    prompts = [d['prompt'] for d in dataset]

    for i in range(0, len(prompts), batch_size):
        batch = model.to_tokens(prompts[i:i+batch_size], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch.to(model.cfg.device),
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        # pattern shape: [batch, heads, seq_len, seq_len]
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        # attention from token_position to all positions
        attn = pattern[:, head, token_position, :].cpu().numpy()
        all_patterns.append(attn)

    return np.concatenate(all_patterns, axis=0).mean(axis=0)


def compute_copy_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    name_list: List[str],
    top_k: int = 5
) -> float:
    """
    Compute copy score 
    
    For each name, compute: residual_stream_at_name_position @ OV @ W_U
    Check if input name appears in top-k logits.
    Returns fraction of names where the head copies correctly.
    """
    W_O = model.W_O[layer, head]       # [d_head, d_model]
    W_V = model.W_V[layer, head]       # [d_model, d_head]
    W_U = model.W_U                    # [d_model, vocab]
    OV  = W_V @ W_O                   # [d_model, d_model]

    correct = 0
    for name in name_list:
        token_id = model.to_single_token(" " + name)
        token_embed = model.W_E[token_id]          # [d_model]
        # Simulate head attending perfectly to this token
        output = token_embed @ OV @ W_U            # [vocab]
        top_tokens = output.topk(top_k).indices.tolist()
        if token_id in top_tokens:
            correct += 1

    return correct / len(name_list)


# Head classification tests 

def test_name_mover(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    name_list: List[str],
    io_position: int = 1,    # position of IO token in prompt
    final_position: int = -1
) -> Dict:
    """
    Name mover signature:
    - At final position, attends to IO token
    """
    attn = get_mean_attention_pattern(
        model, layer, head, dataset, token_position=final_position
    )
    io_attn = float(attn[io_position])
    copy_score = compute_copy_score(model, layer, head, name_list)

    return {
        "io_attention":  io_attn,
        "copy_score":    copy_score,
        "is_name_mover": io_attn > 0.2 and copy_score > 0.5
    }


def test_negative_name_mover(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    name_list: List[str],
    io_position: int = 1,
    final_position: int = -1
) -> Dict:
    """
    Negative name mover: attends to IO but writes in opposite direction.
    Copy score computed with negated OV output.
    """
    attn = get_mean_attention_pattern(
        model, layer, head, dataset, token_position=final_position
    )
    io_attn = float(attn[io_position])

    # Negative copy score: check if name appears in *bottom* k logits
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    W_U = model.W_U
    OV  = W_V @ W_O

    neg_correct = 0
    for name in name_list:
        token_id = model.to_single_token(" " + name)
        token_embed = model.W_E[token_id]
        output = -(token_embed @ OV @ W_U)   # negated
        top_tokens = output.topk(5).indices.tolist()
        if token_id in top_tokens:
            neg_correct += 1
    neg_copy_score = neg_correct / len(name_list)

    return {
        "io_attention":       io_attn,
        "neg_copy_score":     neg_copy_score,
        "is_negative_mover":  io_attn > 0.2 and neg_copy_score > 0.5
    }


def test_s_inhibition(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    s2_position: int = 7,    # position of S2 token
    final_position: int = -1
) -> Dict:
    """
    At final position, attends to S2 token (the repeated subject)
    """
    attn = get_mean_attention_pattern(
        model, layer, head, dataset, token_position=final_position
    )
    s2_attn = float(attn[s2_position])

    return {
        "s2_attention":    s2_attn,
        "is_s_inhibition": s2_attn > 0.2
    }


def test_duplicate_token(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    s1_position: int = 3,   # position of first S occurrence
    s2_position: int = 7    # position of second S occurrence
) -> Dict:

    attn = get_mean_attention_pattern(
        model, layer, head, dataset, token_position=s2_position
    )
    s1_attn = float(attn[s1_position])

    return {
        "s1_from_s2_attention": s1_attn,
        "is_duplicate_token":   s1_attn > 0.2
    }


def test_previous_token(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list
) -> Dict:
    """
    Mean attention on the off-diagonal (position i attends to i-1)
    """
    attn_patterns = []
    prompts = [d['prompt'] for d in dataset]

    for i in range(0, min(100, len(prompts)), 50):
        batch = model.to_tokens(prompts[i:i+50], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch.to(model.cfg.device),
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        attn_patterns.append(pattern[:, head, :, :].cpu())

    patterns = torch.cat(attn_patterns, dim=0)  # [n, seq, seq]
    seq_len = patterns.shape[-1]

    # Off-diagonal: attention from position i to position i-1
    prev_token_scores = []
    for pos in range(1, seq_len):
        prev_token_scores.append(patterns[:, pos, pos-1].mean().item())
    prev_token_score = float(np.mean(prev_token_scores))

    return {
        "prev_token_score":  prev_token_score,
        "is_previous_token": prev_token_score > 0.3
    }


def test_induction(
    model: HookedTransformer,
    layer: int,
    head: int,
    seq_len: int = 50,
    n_seqs: int = 20
) -> Dict:
  
    vocab_size = model.cfg.d_vocab
    half = seq_len // 2

    induction_scores = []
    for _ in range(n_seqs):
        rand_tokens = torch.randint(0, vocab_size, (1, half))
        repeated = torch.cat([rand_tokens, rand_tokens], dim=1).to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                repeated,
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]
        # For each position i in second half, check attention to i-half+1
        score = 0.0
        for pos in range(half, seq_len):
            target = pos - half + 1
            if target < seq_len:
                score += pattern[pos, target].item()
        induction_scores.append(score / half)

    induction_score = float(np.mean(induction_scores))

    return {
        "induction_score":  induction_score,
        "is_induction":     induction_score > 0.3
    }


def classify_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    name_list: List[str],
    io_position: int = 1,
    s1_position: int = 3,
    s2_position: int = 7,
    final_position: int = -1
) -> Dict:
    """
    Run all classification tests on a single head and return
    best functional label with all supporting scores.
    """
    results = {
        "head": (layer, head),
        "name_mover":       test_name_mover(model, layer, head, dataset,
                                             name_list, io_position, final_position),
        "negative_mover":   test_negative_name_mover(model, layer, head, dataset,
                                                      name_list, io_position, final_position),
        "s_inhibition":     test_s_inhibition(model, layer, head, dataset,
                                               s2_position, final_position),
        "duplicate_token":  test_duplicate_token(model, layer, head, dataset,
                                                  s1_position, s2_position),
        "previous_token":   test_previous_token(model, layer, head, dataset),
        "induction":        test_induction(model, layer, head),
    }

    # Determine best label
    label_map = {
        "name_mover":      results["name_mover"]["is_name_mover"],
        "negative_mover":  results["negative_mover"]["is_negative_mover"],
        "s_inhibition":    results["s_inhibition"]["is_s_inhibition"],
        "duplicate_token": results["duplicate_token"]["is_duplicate_token"],
        "previous_token":  results["previous_token"]["is_previous_token"],
        "induction":       results["induction"]["is_induction"],
    }

    matched = [k for k, v in label_map.items() if v]
    results["classification"] = matched if matched else ["unclassified"]

    return results


def classify_circuit(
    model: HookedTransformer,
    circuit_heads: list,
    dataset: list,
    name_list: List[str],
    **kwargs
) -> Dict:

    classifications = {}
    for layer, head in sorted(circuit_heads):
        print(f"  Classifying ({layer}, {head})...")
        classifications[(layer, head)] = classify_head(
            model, layer, head, dataset, name_list, **kwargs
        )
    return classifications