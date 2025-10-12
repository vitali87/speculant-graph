import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loguru import logger
from pyvis.network import Network
from transformers import AutoTokenizer

from speculant_graph import GraphBuilder


def visualize_graph(
    graph_path: str,
    tokenizer_name: str,
    output_html: str = "graph_viz.html",
    hf_token: str | None = None,
    max_nodes: int | None = None,
    min_edge_weight: float = 0.0,
):
    logger.info(f"Loading graph from: {graph_path}")
    graph, metadata = GraphBuilder.load(
        graph_path, validate_tokenizer=True, expected_tokenizer=tokenizer_name
    )

    logger.info("Graph metadata:")
    logger.info(f"  Tokenizer: {metadata['tokenizer_name']}")
    logger.info(f"  Nodes: {metadata['num_nodes']}")
    logger.info(f"  Edges: {metadata['num_edges']}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)

    if max_nodes and graph.number_of_nodes() > max_nodes:
        logger.info(f"Limiting visualization to top {max_nodes} most frequent tokens")

        token_nodes = [
            (node, data)
            for node, data in graph.nodes(data=True)
            if isinstance(node, int)
        ]
        token_nodes_sorted = sorted(
            token_nodes, key=lambda x: x[1].get("count", 0), reverse=True
        )[:max_nodes]
        selected_tokens = {node for node, _ in token_nodes_sorted}

        context_nodes = set()
        for from_node, to_node, edge_data in graph.edges(data=True):
            if to_node in selected_tokens and isinstance(from_node, tuple):
                weight = edge_data.get("weight", 0)
                if weight >= min_edge_weight:
                    context_nodes.add(from_node)

        selected_nodes = selected_tokens | context_nodes
        logger.info(
            f"Selected {len(selected_tokens)} tokens and {len(context_nodes)} context nodes"
        )
        subgraph = graph.subgraph(selected_nodes).copy()
    else:
        subgraph = graph

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
    )

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=100,
        spring_strength=0.001,
        damping=0.09,
    )

    logger.info("Adding nodes to visualization...")
    for node_id, data in subgraph.nodes(data=True):
        if isinstance(node_id, tuple):
            token_text = tokenizer.decode(list(node_id))
            order = len(node_id)
            count = 0
            display_text = f"O{order}: {token_text}"
            node_str = str(node_id)
        else:
            token_text = tokenizer.decode([node_id])
            count = data.get("count", 0)
            display_text = token_text
            node_str = str(node_id)

        display_text = display_text.replace("\n", "\\n").replace("\t", "\\t")
        if len(display_text) > 20:
            display_text = display_text[:20] + "..."

        size = min(10 + (count * 2), 50) if isinstance(node_id, int) else 15

        if isinstance(node_id, tuple):
            title = f"Order-{order} context\nTokens: {node_id}\nText: '{token_text}'"
        else:
            title = f"Token ID: {node_id}\nText: '{token_text}'\nCount: {count}"

        net.add_node(
            node_str,
            label=display_text,
            title=title,
            size=size,
            color="#00ff41" if isinstance(node_id, int) else "#ff9900",
        )

    logger.info("Adding edges to visualization...")
    edge_count = 0
    for from_node, to_node, data in subgraph.edges(data=True):
        weight = data.get("weight", 0)
        count = data.get("count", 0)

        if weight < min_edge_weight:
            continue

        if isinstance(from_node, tuple):
            from_text = tokenizer.decode(list(from_node))
        else:
            from_text = tokenizer.decode([from_node])

        if isinstance(to_node, tuple):
            to_text = tokenizer.decode(list(to_node))
        else:
            to_text = tokenizer.decode([to_node])

        order = data.get("order", 1)
        title = f"Order-{order}\n'{from_text}' → '{to_text}'\nProb: {weight:.3f}\nCount: {count}"

        edge_width = max(0.5, weight * 10)

        net.add_edge(
            str(from_node), str(to_node), title=title, width=edge_width, color="#888888"
        )
        edge_count += 1

    logger.info(
        f"Visualization contains {len(subgraph.nodes())} nodes and {edge_count} edges"
    )

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 200
            }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        },
        "edges": {
            "smooth": {
                "type": "curvedCW",
                "roundness": 0.2
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        }
    }
    """)

    logger.info(f"Saving visualization to: {output_html}")
    net.save_graph(output_html)
    logger.info(f"✓ Open {output_html} in your browser to view the graph")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize knowledge graph")
    parser.add_argument("graph_path", help="Path to the graph .pkl file")
    parser.add_argument(
        "--tokenizer", default="meta-llama/Llama-3.2-3B", help="Tokenizer name"
    )
    parser.add_argument("--output", default="graph_viz.html", help="Output HTML file")
    parser.add_argument("--max-nodes", type=int, help="Limit to N most frequent tokens")
    parser.add_argument(
        "--min-weight", type=float, default=0.0, help="Minimum edge probability"
    )
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    visualize_graph(
        args.graph_path,
        args.tokenizer,
        args.output,
        hf_token,
        args.max_nodes,
        args.min_weight,
    )


if __name__ == "__main__":
    main()
