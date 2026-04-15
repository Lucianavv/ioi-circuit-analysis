

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Dict, Tuple, Optional
from src.attention_analysis import get_attention_pattern
from src.patching import path_patch_head


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


def compute_attention_scores(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    batch_size: int = 50
) -> dict:
  
    io_attns, s2_attns, dup_attns, prev_attns = [], [], [], []

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
        seq_len = pattern.shape[-1]

        for j, prompt_data in enumerate(batch):
            pos = find_token_positions(model, prompt_data)

            if pos["io"] is not None and pos["end"] is not None:
                io_attns.append(pattern[j, head, pos["end"], pos["io"]].item())

            if pos["s2"] is not None and pos["end"] is not None:
                s2_attns.append(pattern[j, head, pos["end"], pos["s2"]].item())

            if pos["s1"] is not None and pos["s2"] is not None:
                dup_attns.append(pattern[j, head, pos["s2"], pos["s1"]].item())

            # Previous token: mean off-diagonal
            prev = [pattern[j, head, p, p-1].item() for p in range(1, seq_len)]
            prev_attns.append(np.mean(prev))

    return {
        "io_attn":        float(np.mean(io_attns))   if io_attns  else 0.0,
        "s2_attn":        float(np.mean(s2_attns))   if s2_attns  else 0.0,
        "duplicate_attn": float(np.mean(dup_attns))  if dup_attns else 0.0,
        "prev_token_attn": float(np.mean(prev_attns)) if prev_attns else 0.0,
    }


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

        score = sum(pattern[pos, pos-half+1].item() 
                   for pos in range(half, seq_len) 
                   if pos-half+1 < seq_len) / half
        scores.append(score)

    return float(np.mean(scores))



def compute_dla(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    batch_size: int = 50
) -> float:

    dla_scores = []

    for i in range(0, len(dataset), batch_size):
        batch   = dataset[i:i+batch_size]
        prompts = [d["prompt"] for d in batch]
        tokens  = model.to_tokens(prompts, prepend_bos=True).to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer}.attn.hook_result"
            )
        # hook_result: [batch, seq, n_heads, d_model]
        head_out = cache[f"blocks.{layer}.attn.hook_result"][:, -1, head, :]

        for j, prompt_data in enumerate(batch):
            io_token = model.to_single_token(" " + prompt_data["correct_answer"])
            s_token  = model.to_single_token(" " + prompt_data["incorrect_answer"])

            # Logit difference direction in vocab space
            logit_diff_dir = model.W_U[:, io_token] - model.W_U[:, s_token]
            logit_diff_dir = logit_diff_dir.to(model.cfg.device)

            dla = (head_out[j] @ logit_diff_dir).item()
            dla_scores.append(dla)

    return float(np.mean(dla_scores))


def compute_patching_effect(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    n_samples: int = 100
) -> float:

    from src.metrics import compute_logit_difference

    recoveries = []
    for d in dataset[:n_samples]:
        clean_tokens     = model.to_tokens(d["prompt"], prepend_bos=True)
        corrupted_tokens = model.to_tokens(d["corrupted_prompt"], prepend_bos=True)

        clean_ld, _, _ = compute_logit_difference(
            model, d["prompt"], d["correct_answer"], d["incorrect_answer"]
        )

        patched_logits = path_patch_head(
            model, clean_tokens, corrupted_tokens, layer, head
        )
        patched_ld = (
            patched_logits[0, -1, model.to_single_token(" " + d["correct_answer"])]
          - patched_logits[0, -1, model.to_single_token(" " + d["incorrect_answer"])]
        ).item()

        if abs(clean_ld) > 0.1:
            recoveries.append(patched_ld / clean_ld)

    return float(np.mean(recoveries)) if recoveries else 0.0



def classify_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    dataset: list,
    thresholds: dict = None,
    run_patching: bool = False
) -> dict:
  
    if thresholds is None:
        thresholds = {
            "io_attn":       0.15,
            "s2_attn":       0.15,
            "duplicate_attn": 0.15,
            "prev_token":    0.30,
            "induction":     0.30,
            "dla_positive":  0.10,   # DLA > threshold - name mover
            "dla_negative":  -0.10,  # DLA < threshold - negative mover
        }

    attn  = compute_attention_scores(model, layer, head, dataset)
    dla   = compute_dla(model, layer, head, dataset)
    ind   = compute_induction_score(model, layer, head)
    patch = compute_patching_effect(model, layer, head, dataset) if run_patching else None

    labels = []
    if attn["io_attn"] > thresholds["io_attn"] and dla > thresholds["dla_positive"]:
        labels.append("name_mover")
    if attn["io_attn"] > thresholds["io_attn"] and dla < thresholds["dla_negative"]:
        labels.append("negative_mover")
    if attn["s2_attn"] > thresholds["s2_attn"]:
        labels.append("s_inhibition")
    if attn["duplicate_attn"] > thresholds["duplicate_attn"]:
        labels.append("duplicate_token")
    if attn["prev_token_attn"] > thresholds["prev_token"]:
        labels.append("previous_token")
    if ind > thresholds["induction"]:
        labels.append("induction")

    return {
        "head":             (layer, head),
        "io_attn":          attn["io_attn"],
        "s2_attn":          attn["s2_attn"],
        "duplicate_attn":   attn["duplicate_attn"],
        "prev_token_attn":  attn["prev_token_attn"],
        "induction_score":  ind,
        "dla":              dla,
        "patching_effect":  patch,
        "classification":   labels if labels else ["unclassified"]
    }


def classify_circuit(
    model: HookedTransformer,
    circuit_heads: list,
    dataset: list,
    thresholds: dict = None,
    run_patching: bool = False
) -> dict:

    results = {}
    total = len(circuit_heads)
    for i, (layer, head) in enumerate(sorted(circuit_heads)):
        print(f"  [{i+1}/{total}] ({layer},{head})...", end=" ")
        results[(layer, head)] = classify_head(
            model, layer, head, dataset, thresholds, run_patching
        )
        print(f"→ {results[(layer,head)]['classification']}")
    return results
