# 🧠 IOI Circuit Analysis

> *Do the same circuits that make GPT-2 understand indirect objects also exist in GPT-Neo? Let's find out.*

🎓 Bachelor's thesis in Computer Science — Universidad San Francisco de Quito, 2025  

---

## What is this?

This project investigates whether the **Indirect Object Identification (IOI) circuit** — a specific set of attention heads discovered in GPT-2 Small by Wang et al. (2022) — also emerges in GPT-Neo 125M. This is a test of one of mechanistic interpretability's biggest open questions: do circuits *generalize* across architectures?

Two automated circuit discovery methods are compared head-to-head:
- 🔍 **ACDC** — greedy edge search, thorough but slow (Conmy et al., 2023)
- ⚡ **EAP** — gradient-based attribution, fast approximation (Syed et al., 2023)

---

## 📊 Results at a Glance

| | GPT-2 Small · ACDC | GPT-2 Small · EAP | GPT-Neo · ACDC | GPT-Neo · EAP |
|---|---|---|---|---|
| Circuit size | 57 heads | 14 heads | 123 heads | 32 heads (minimal) |
| Known heads recovered | 15/17 · 88% | 8/17 · 47% | 19/26 · 73% | 8/26 · 31% |
| Faithfulness | 1.898 | 0.132 | 0.930 | 1.302 |
| Forward passes | ~10,200 | ~200 | ~thousands | ~300 |

**TL;DR:** Structurally analogous heads appear at the same layer positions in both models (9.9, 8.6, 5.8, 0.10 show up in GPT-Neo too 👀). ACDC recovers 100% of S-inhibition and duplicate token head classes. The evidence points toward *functional* universality — the circuit exists, but it's not a perfect structural copy.

---

## 🗂️ Repo Structure

```
ioi-circuit-analysis/
├── src/                       # clean, reusable modules
│   ├── ioi_dataset.py         # dataset generation + pABC baseline
│   ├── metrics.py             # logit difference, faithfulness, sparsity
│   ├── patching.py            # activation patching
│   ├── acdc.py                # head-level ACDC
│   ├── attribution_patching.py
│   ├── attention_analysis.py  # DLA + attention patterns
│   └── validation.py          # known head recovery
│
├── notebooks/                 # all Colab experiments, in order
├── data/
│   ├── prompts/               # IOI + pABC datasets
│   └── results/               # saved tensors + metrics
├── figures/                   # circuit graphs + heatmaps
└── requirements.txt
```

---

## 📓 Notebooks

| Notebook | What it does |
|---|---|
| `Hello_IOI.ipynb` | First contact: attention visualization, logit difference, induction heads |
| `e1_setup.ipynb` | Full setup: tokenization, dataset generation, patching built from scratch |
| `e2.ipynb` | GPT-2 Small experiments: ACDC + EAP, threshold sweep, faithfulness debugging |
| `abc.ipynb` | Builds the pABC corrupted reference dataset for mean ablation |
| `e3.ipynb` | GPT-Neo EAP: full edge-level run, Pareto sweep, minimality → 110 edges / 32 heads |
| `EAP_components_analysis.ipynb` | Head classification, DLA heatmaps, circuit graph for GPT-Neo EAP |
| `acdc.ipynb` | GPT-Neo ACDC via original paper repo, tau sweep, 123-head circuit |

All notebooks are self-contained — they clone the repo and install dependencies themselves. Just open in Colab and run top to bottom.

> ⚠️ **numpy note:** The first cell of most notebooks installs `numpy==1.26` and restarts the runtime. This is required — TransformerLens breaks with numpy 2.x. Don't skip it.

---

## ⚙️ Model Config for EAP

When loading either model for edge-level EAP, you need these flags before `model.setup()`:

```python
model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
model.setup()
```

This enables the hook points for the full EAP computational graph (score matrix shape: [168, 456] for GPT-Neo).

---

## 📚 References

- Wang et al. (2022) — [Interpretability in the Wild](https://arxiv.org/abs/2211.00593)
- Conmy et al. (2023) — [Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997)
- Syed et al. (2023) — [Attribution Patching Outperforms ACDC](https://arxiv.org/abs/2310.10348)
- Elhage et al. (2021) — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda
