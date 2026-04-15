

import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Optional, Set



NODE_COLORS = {
    "head":  "#F4A460",   
    "mlp":   "#87CEEB",  
    "resid": "#BD8AD0",   
    "head_q": "#FFEA70",  
    "head_k": "#FFA07A",  
    "head_v": "#B2DBBF",  

    "name_mover":      "#EB6A5C",  
    "negative_mover":  "#8E44AD",  
    "s_inhibition":    "#16A085",  
    "induction":       "#2980B9",  
    "duplicate_token": "#7EC97E", 
    "previous_token":  "#FFD6E0",  
    "unclassified":    "#BDC3C7",  
}



def _parse_node(node_name: str):

    parts = node_name.split(".")
    if node_name.startswith("head"):
        layer, head = int(parts[1]), int(parts[2])
        base = f"head.{layer}.{head}"
        qkv  = parts[3] if len(parts) == 4 else None
        return base, f"head_{qkv}" if qkv else "head", layer, head, qkv
    elif node_name.startswith("mlp"):
        layer = int(parts[1])
        return node_name, "mlp", layer, None, None
    else:
        layer = int(parts[-1]) if parts[-1].isdigit() else 0
        return node_name, "resid", layer, None, None


def _node_color(node_type: str, base_name: str,
                head_classifications: Optional[Dict]) -> str:

    if head_classifications and node_type == "head":
        parts = base_name.split(".")
        if len(parts) == 3:
            key = (int(parts[1]), int(parts[2]))
            if key in head_classifications:
                label = head_classifications[key]
                return NODE_COLORS.get(label, NODE_COLORS["unclassified"])
    return NODE_COLORS.get(node_type, "#DDDDDD")


def draw_circuit(
    edges: List[Tuple],
    output_path: str = "circuit",
    title: str = "",
    head_classifications: Optional[Dict] = None,
    min_score: Optional[float] = None,
    show_scores: bool = False,
    format: str = "png"
):
  
    if min_score is not None:
        edges = [(f, t, s) for f, t, s in edges if abs(s) >= min_score]

    dot = graphviz.Digraph(
        comment=title,
        graph_attr={
            'rankdir':  'TB',
            'ranksep':  '0.6',
            'nodesep':  '0.35',
            'splines':  'curved',
            'fontname': 'Helvetica',
            'bgcolor':  'white',
        },
        node_attr={
            'style':    'filled,rounded',
            'shape':    'box',
            'fontname': 'Helvetica',
            'fontsize': '9',
            'margin':   '0.08,0.04',
            'penwidth': '1.2',
        },
        edge_attr={
            'arrowsize': '0.5',
            'fontsize':  '7',
        }
    )

    # Collect nodes grouped by layer
    nodes_seen = {}
    for fn, tn, score in edges:
        for n in [fn, tn]:
            if n not in nodes_seen:
                nodes_seen[n] = _parse_node(n)

    layer_nodes = {}
    for node_name, parsed in nodes_seen.items():
        base, ntype, layer, head, qkv = parsed
        layer_nodes.setdefault(layer, []).append((node_name, base, ntype, head, qkv))

    # Add nodes in layer subgraphs for alignment
    for layer in sorted(layer_nodes.keys()):
        with dot.subgraph(name=f"cluster_layer_{layer}") as sub:
            sub.attr(rank='same', style='invis')
            for node_name, base, ntype, head, qkv in layer_nodes[layer]:
                color = _node_color(ntype, base, head_classifications)
                if ntype.startswith("head"):
                    label = f"a{layer}.{head}_{qkv}" if qkv else f"a{layer}.{head}"
                elif ntype == "mlp":
                    label = f"m{layer}"
                else:
                    prefix = "r↑" if "pre" in node_name else "r↓"
                    label  = f"{prefix}{layer}"
                sub.node(node_name, label=label, fillcolor=color, color="#555555")

    # Add edges — color matches source node type
    for fn, tn, score in edges:
        _, ntype, _, _, _ = _parse_node(fn)
        base_color = NODE_COLORS.get(ntype, "#555555")
        edge_style = "dashed" if score < 0 else "solid"
        penwidth   = str(max(0.5, min(4.0, abs(score) * 15)))
        label      = f"{score:.3f}" if show_scores else ""

        dot.edge(fn, tn,
                 label=label,
                 style=edge_style,
                 penwidth=penwidth,
                 color=base_color,
                 arrowhead="vee")

    dot.render(output_path, format=format, cleanup=True)
    print(f"Circuit saved: {output_path}.{format}")
    return dot


# Head heatmaps 

def plot_head_heatmap(
    score_matrix: np.ndarray,
    circuit_heads: Set[Tuple[int, int]],
    output_path: str,
    colormap: str = "PRGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    cbar_label: str = "",
    figsize: Tuple = (5.5, 5.0)
):
   
    fig, ax = plt.subplots(figsize=figsize)

    if center is not None:
        abs_max = max(abs(float(score_matrix.max())),
                      abs(float(score_matrix.min())), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=center, vmax=abs_max)
        im = ax.imshow(score_matrix, cmap=colormap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(score_matrix, cmap=colormap,
                       vmin=vmin, vmax=vmax, aspect='auto')

    # Mark circuit heads
    for (layer, head) in circuit_heads:
        if 0 <= layer < 12 and 0 <= head < 12:
            ax.plot(head, layer, 'o',
                    color='white', markersize=7,
                    markeredgecolor='black', markeredgewidth=1.2,
                    zorder=5)

    ax.set_xlabel('Head', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(range(12), fontsize=8)
    ax.set_yticklabels(range(12), fontsize=8)
    ax.tick_params(length=0)

    # Clean border
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#888888')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap_suite(
    all_classifications: Dict,
    circuit_heads: Set[Tuple[int, int]],
    output_dir: str = "figures",
    prefix: str = "neo_eap"
):
   
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Build 12x12 matrices
    io_mat  = np.zeros((12, 12))
    s2_mat  = np.zeros((12, 12))
    dla_mat = np.zeros((12, 12))
    ind_mat = np.zeros((12, 12))
    dup_mat = np.zeros((12, 12))
    prev_mat = np.zeros((12, 12))

    for (layer, head), r in all_classifications.items():
        io_mat[layer][head]   = r.get('io_attn', 0)
        s2_mat[layer][head]   = r.get('s2_attn', 0)
        dla_mat[layer][head]  = r.get('dla', 0)
        ind_mat[layer][head]  = r.get('induction_score', 0)
        dup_mat[layer][head]  = r.get('duplicate_attn', 0)
        prev_mat[layer][head] = r.get('prev_token_attn', 0)

    # IO attention
    plot_head_heatmap(
        io_mat, circuit_heads,
        output_path=f"{output_dir}/{prefix}_heatmap_io_attn.png",
        colormap='Greens', vmin=0,
        cbar_label='Mean IO attention at END position'
    )

    # DLA 
    plot_head_heatmap(
        dla_mat, circuit_heads,
        output_path=f"{output_dir}/{prefix}_heatmap_dla.png",
        colormap='RdBu', center=0,
        cbar_label='Direct Logit Attribution (IO - S)'
    )

    # S2 attention 
    plot_head_heatmap(
        s2_mat, circuit_heads,
        output_path=f"{output_dir}/{prefix}_heatmap_s2_attn.png",
        colormap='Greens', vmin=0,
        cbar_label='Mean S2 attention at END position'
    )

    # Induction score 
    plot_head_heatmap(
        ind_mat, circuit_heads,
        output_path=f"{output_dir}/{prefix}_heatmap_induction.png",
        colormap='Greens', vmin=0,
        cbar_label='Induction score (repeated sequences)'
    )

    print(f"All heatmaps saved to {output_dir}/")
    return {
        'io_matrix':  io_mat,
        'dla_matrix': dla_mat,
        's2_matrix':  s2_mat,
        'ind_matrix': ind_mat,
        'dup_matrix': dup_mat,
        'prev_matrix': prev_mat,
    }
