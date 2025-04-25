import torch
from torch_geometric.data import Data

def build_edge_index(nodes, id_map):
    """
    Given a list of nodes (with 'id' and 'parent_id'), build edge_index for reply graph.
    Only includes valid edges where both parent and child exist in id_map.
    """
    edges = []
    for node in nodes:
        child_id = node['id']
        parent_id = node.get('parent_id')

        if parent_id is None:
            continue

        # Clean ID formats (e.g., t1_xxx or t3_xxx)
        parent_id = parent_id.replace('t1_', '').replace('t3_', '')

        if parent_id in id_map and child_id in id_map:
            parent_idx = id_map[parent_id]
            child_idx = id_map[child_id]
            edges.append([parent_idx, child_idx])

    if len(edges) == 0:
        # fallback: avoid crash on empty thread
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def thread_to_graph(nodes, label_map):
    """
    Converts a thread (submission + comments) into a torch_geometric.data.Data object.
    Each node should contain: 'id', 'text', 'parent_id' (optional)
    """
    texts = [node['text'] for node in nodes]
    ids = [node['id'] for node in nodes]
    id_map = {node_id: idx for idx, node_id in enumerate(ids)}

    edge_index = build_edge_index(nodes, id_map)

    graph = Data(
        x=texts,  # raw text to be encoded by BERT later
        edge_index=edge_index,
        y=torch.tensor([label_map[nodes[0]['subreddit']]], dtype=torch.long)
    )
    return graph
