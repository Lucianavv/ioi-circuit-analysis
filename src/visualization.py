
import graphviz
from typing import List, Tuple, Dict, Optional

NODE_COLORS = {
    "head":            "#FFB347",  
    "head_q":          "#FFD700",  # query input
    "head_k":          "#FFA07A",  # key input  
    "head_v":          "#FF8C00",  # value input
    "mlp":             "#87CEEB",  
    "resid":           "#90EE90",  
    # functional type colors 
    "name_mover":      "#FF6B6B",
    "negative_mover":  "#C0392B",
    "s_inhibition":    "#4ECDC4",
    "induction":       "#45B7D1",
    "duplicate_token": "#96CEB4",
    "previous_token":  "#FFEAA7",
    "unclassified":    "#DDA0DD",
}

def parse_node(node_name: str):
    """
    Parse EAP node name into (base_name, node_type, layer, head, qkv).
    Examples:
      'head.9.4.v'    base='head.9.4', type='head_v',  layer=9,  head=4
      'head.9.4'      base='head.9.4', type='head',    layer=9,  head=4
      'mlp.5'         base='mlp.5',    type='mlp',     layer=5
    """
    parts = node_name.split(".")
    if node_name.startswith("head"):
        layer = int(parts[1])
        head  = int(parts[2])
        base  = f"head.{layer}.{head}"
        if len(parts) == 4:
            qkv = parts[3]
            return base, f"head_{qkv}", layer, head, qkv
        return base, "head", layer, head, None
    elif node_name.startswith("mlp"):
        layer = int(parts[1])
        return node_name, "mlp", layer, None, None
    else:
        # resid_pre or resid_post
        layer = int(parts[-1]) if parts[-1].isdigit() else 0
        return node_name, "resid", layer, None, None


def get_node_color(node_type: str,
                   base_name: str,
                   head_classifications: Optional[Dict]) -> str:

    #Only full head nodes (not q/k/v inputs) get classification colors.

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
    title: str = "IOI Circuit",
    head_classifications: Optional[Dict] = None,
    min_score: Optional[float] = None,
    show_scores: bool = False,
    format: str = "png"
) -> graphviz.Digraph:
 
    if min_score is not None:
        edges = [(f, t, s) for f, t, s in edges if abs(s) >= min_score]

    dot = graphviz.Digraph(
        comment=title,
        graph_attr={
            'rankdir':  'TB',
            'ranksep':  '0.8',
            'nodesep':  '0.4',
            'fontname': 'Helvetica',
            'label':    title,
            'fontsize': '16',
            'splines':  'ortho',
        },
        node_attr={
            'style':    'filled,rounded',
            'shape':    'box',
            'fontname': 'Helvetica',
            'fontsize': '9',
            'margin':   '0.1,0.05',
        },
        edge_attr={
            'arrowsize': '0.5',
            'fontsize':  '7',
        }
    )

    # Collect all nodes and group by layer 
    nodes_seen = {}  # node_name - (base, node_type, layer, head, qkv)
    for from_node, to_node, score in edges:
        for n in [from_node, to_node]:
            if n not in nodes_seen:
                nodes_seen[n] = parse_node(n)

    # Group by layer for subgraph layout
    layer_nodes = {}
    for node_name, (base, ntype, layer, head, qkv) in nodes_seen.items():
        layer_nodes.setdefault(layer, []).append((node_name, base, ntype, head, qkv))

    # Add nodes grouped into layer subgraphs 
    for layer in sorted(layer_nodes.keys()):
        with dot.subgraph(name=f"cluster_layer_{layer}") as sub:
            sub.attr(
                rank='same',
                style='invis',
                label=f"Layer {layer}"
            )
            for node_name, base, ntype, head, qkv in layer_nodes[layer]:
                color = get_node_color(ntype, base, head_classifications)
                # Label: show short name
                if ntype.startswith("head"):
                    if qkv:
                        label = f"a{layer}.{head}_{qkv}"
                    else:
                        label = f"a{layer}.{head}"
                elif ntype == "mlp":
                    label = f"m{layer}"
                else:
                    prefix = "r_pre" if "pre" in node_name else "r_post"
                    label = f"{prefix}.{layer}"

                sub.node(
                    node_name,
                    label=label,
                    fillcolor=color,
                    color="#444444",
                    penwidth="1.2"
                )

    # Add implicit head output nodes 
    head_outputs_needed = set()
    for from_node, to_node, score in edges:
        base, ntype, layer, head, qkv = parse_node(from_node)
        if ntype == "head" and qkv is None:
            head_outputs_needed.add(from_node)

    for node_name in head_outputs_needed:
        if node_name not in nodes_seen:
            base, ntype, layer, head, qkv = parse_node(node_name)
            color = get_node_color("head", base, head_classifications)
            dot.node(node_name,
                     label=f"a{layer}.{head}",
                     fillcolor=color,
                     color="#444444",
                     style="filled,rounded",
                     shape="box")

    # Add edges 
    for from_node, to_node, score in edges:
        edge_style   = "dashed" if score < 0 else "solid"
        penwidth     = str(max(0.5, min(4.0, abs(score) * 15)))
        edge_color   = "#CC3333" if score < 0 else "#333333"
        label        = f"{score:.3f}" if show_scores else ""

        dot.edge(
            from_node, to_node,
            label=label,
            style=edge_style,
            penwidth=penwidth,
            color=edge_color,
            arrowhead="vee"
        )

    dot.render(output_path, format=format, cleanup=True)
    print(f"Saved: {output_path}.{format}")
    return dot


def draw_legend(output_path: str = "circuit_legend"):
    #Generate a standalone color legend for circuit graphs.
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'label': 'Legend'})
    for label, color in NODE_COLORS.items():
        dot.node(label, fillcolor=color, style='filled,rounded',
                 shape='box', fontname='Helvetica', fontsize='10')
    dot.render(output_path, format='png', cleanup=True)
    print(f"Legend saved: {output_path}.png")