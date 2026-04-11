

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Tuple, List, Dict, Optional


def find_token_positions(model: HookedTransformer, prompt_data: dict) -> dict:
    """
    Find S1, S2, IO, END positions for a prompt.
    Template-agnostic — works for all three IOI templates.
    Returns dict with keys: io, s1, s2, end (None if not found).
    """
    tokens = model.to_tokens(prompt_data["prompt"], prepend_bos=True)[0]
    token_list = tokens.tolist()

    io_id = model.to_single_token(" " + prompt_data["correct_answer"])
    s_id  = model.to_single_token(" " + prompt_data["incorrect_answer"])

    s_positions  = [i for i, t in enumerate(token_list) if t == s_id]
    io_positions = [i for i, t in enumerate(token_list) if t == io_id]

    return {
        "io":  io_positions[0] if io_positions else None,
        "s1":  s_positions[0]  if len(s_positions) > 0 else None,
        "s2":  s_positions[1]  if len(s_positions) > 1 else None,
        "end": len(token_list) - 1
    }


def get_mean_attention_pattern(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    source_key: str = "end",
    target_key: str = "io",
    batch_size: int = 50
) -> float:
    """
    Compute mean attention from source_key position to target_key position
    across a dataset, using dynamic token position lookup per prompt.

    source_key, target_key: keys from find_token_positions output
    ('io', 's1', 's2', 'end')
    Returns mean attention probability.
    """
    attention_values = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [d["prompt"] for d in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens.to(model.cfg.device),
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"].cpu()

        for j, prompt_data in enumerate(batch):
            pos = find_token_positions(model, prompt_data)
            src = pos[source_key]
            tgt = pos[target_key]
            if src is not None and tgt is not None:
                attn_val = pattern[j, head, src, tgt].item()
                attention_values.append(attn_val)

    return float(np.mean(attention_values)) if attention_values else 0.0


def compute_copy_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    name_list: List[str],
    top_k: int = 5
) -> float:
    """
    Copy score: fraction of names where head's OV matrix
    places the input name in top-k output logits.
    """
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    W_U = model.W_U
    OV  = W_V @ W_O

    correct = 0
    for name in name_list:
        token_id   = model.to_single_token(" " + name)
        token_embed = model.W_E[token_id]
        output      = token_embed @ OV @ W_U
        top_tokens  = output.topk(top_k).indices.tolist()
        if token_id in top_tokens:
            correct += 1

    return correct / len(name_list)


def compute_neg_copy_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    name_list: List[str],
    top_k: int = 5
) -> float:
    """
    Negative copy score: checks if head writes in opposite direction.
    """
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    W_U = model.W_U
    OV  = W_V @ W_O

    correct = 0
    for name in name_list:
        token_id    = model.to_single_token(" " + name)
        token_embed = model.W_E[token_id]
        output      = -(token_embed @ OV @ W_U)
        top_tokens  = output.topk(top_k).indices.tolist()
        if token_id in top_tokens:
            correct += 1

    return correct / len(name_list)


def compute_induction_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    seq_len: int = 50,
    n_seqs: int = 20
) -> float:
   
    vocab_size = model.cfg.d_vocab
    half = seq_len // 2
    scores = []

    for _ in range(n_seqs):
        rand_tokens = torch.randint(0, vocab_size, (1, half))
        repeated    = torch.cat([rand_tokens, rand_tokens], dim=1).to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                repeated,
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].cpu()

        score = 0.0
        for pos in range(half, seq_len):
            target = pos - half + 1
            if target < seq_len:
                score += pattern[pos, target].item()
        scores.append(score / half)

    return float(np.mean(scores))


def compute_prev_token_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    n_samples: int = 100,
    batch_size: int = 50
) -> float:
    """
    Previous token score: mean off-diagonal attention (pos i → pos i-1).
    """
    scores = []
    for i in range(0, min(n_samples, len(dataset)), batch_size):
        batch   = dataset[i:i+batch_size]
        prompts = [d["prompt"] for d in batch]
        tokens  = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens.to(model.cfg.device),
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"].cpu()
        seq_len = pattern.shape[-1]

        for b in range(pattern.shape[0]):
            prev = [pattern[b, head, pos, pos-1].item()
                    for pos in range(1, seq_len)]
            scores.append(np.mean(prev))

    return float(np.mean(scores))


def compute_duplicate_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    batch_size: int = 50
) -> float:
    """
    Duplicate token score: from S2 position, mean attention to S1 position.
    """
    scores = []
    for i in range(0, len(dataset), batch_size):
        batch   = dataset[i:i+batch_size]
        prompts = [d["prompt"] for d in batch]
        tokens  = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens.to(model.cfg.device),
                names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"].cpu()

        for j, prompt_data in enumerate(batch):
            pos = find_token_positions(model, prompt_data)
            if pos["s1"] is not None and pos["s2"] is not None:
                scores.append(pattern[j, head, pos["s2"], pos["s1"]].item())

    return float(np.mean(scores)) if scores else 0.0


def classify_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    name_list: List[str],
    thresholds: dict = None
) -> dict:
    """
    Run all classification tests on a single head.
    Returns scores and best functional label.
    """
    if thresholds is None:
        thresholds = {
            "io_attn":       0.15,
            "s2_attn":       0.15,
            "copy_score":    0.50,
            "neg_copy":      0.50,
            "induction":     0.30,
            "prev_token":    0.30,
            "duplicate":     0.15,
        }

    # Compute all scores
    io_attn    = get_mean_attention_pattern(model, layer, head, dataset,
                                            source_key="end", target_key="io")
    s2_attn    = get_mean_attention_pattern(model, layer, head, dataset,
                                            source_key="end", target_key="s2")
    s1_from_s2 = compute_duplicate_score(model, layer, head, dataset)
    copy       = compute_copy_score(model, layer, head, name_list)
    neg_copy   = compute_neg_copy_score(model, layer, head, name_list)
    induction  = compute_induction_score(model, layer, head)
    prev_token = compute_prev_token_score(model, layer, head, dataset)

    # Classification logic
    labels = []
    if io_attn > thresholds["io_attn"] and copy > thresholds["copy_score"]:
        labels.append("name_mover")
    if io_attn > thresholds["io_attn"] and neg_copy > thresholds["neg_copy"]:
        labels.append("negative_mover")
    if s2_attn > thresholds["s2_attn"]:
        labels.append("s_inhibition")
    if s1_from_s2 > thresholds["duplicate"]:
        labels.append("duplicate_token")
    if prev_token > thresholds["prev_token"]:
        labels.append("previous_token")
    if induction > thresholds["induction"]:
        labels.append("induction")

    return {
        "head":           (layer, head),
        "io_attn":        io_attn,
        "s2_attn":        s2_attn,
        "duplicate_score": s1_from_s2,
        "copy_score":     copy,
        "neg_copy_score": neg_copy,
        "induction_score": induction,
        "prev_token_score": prev_token,
        "classification": labels if labels else ["unclassified"]
    }


def classify_circuit(
    model: HookedTransformer,
    circuit_heads: list,
    dataset: list,
    name_list: List[str],
    thresholds: dict = None
) -> dict:

    results = {}
    total = len(circuit_heads)
    for i, (layer, head) in enumerate(sorted(circuit_heads)):
        print(f"  [{i+1}/{total}] Classifying ({layer},{head})...")
        results[(layer, head)] = classify_head(
            model, layer, head, dataset, name_list, thresholds
        )
    return results
