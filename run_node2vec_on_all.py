import torch
from pathlib import Path
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from tqdm import tqdm

from constants import (
    NODE2VEC_DIM,
    NODE2VEC_WALK_LENGTH,
    NODE2VEC_CONTEXT_SIZE,
    NODE2VEC_NUM_WALKS,
    NODE2VEC_P,
    NODE2VEC_Q,
    DEVICE,
    TRAIN_GRAPH_PATH,
    VAL_GRAPH_PATH,
    TEST_GRAPH_PATH
)



def run_node2vec(graphs, dim=NODE2VEC_DIM):
    processed_graphs = []

    for graph in tqdm(graphs, desc="Embedding graphs with Node2Vec"):
        if graph.edge_index.size(1) == 0:
            # Skip graphs with no edges
            graph.struct_emb = torch.zeros((graph.num_nodes, dim))
            processed_graphs.append(graph)
            continue

        node2vec = Node2Vec(
            edge_index=graph.edge_index,
            embedding_dim=dim,
            walk_length=NODE2VEC_WALK_LENGTH,
            context_size=NODE2VEC_CONTEXT_SIZE,
            walks_per_node=NODE2VEC_NUM_WALKS,
            p=NODE2VEC_P,
            q=NODE2VEC_Q,
            sparse=True
        ).to(DEVICE)

        loader = node2vec.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

        node2vec.train()
        for epoch in range(1):  # Just one epoch per graph (small graphs)
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            emb = node2vec.embedding.weight.detach().cpu()
            graph.struct_emb = emb

        processed_graphs.append(graph)

    return processed_graphs


def process_and_save(path: Path):
    print(f"\n📦 Loading: {path.name}")
    graphs = torch.load(path)  # No need for weights_only=False here in 2.1.0
    processed = run_node2vec(graphs)
    torch.save(processed, path)
    print(f"✅ Saved updated graphs with struct_emb → {path.name}")



if __name__ == "__main__":
    process_and_save(Path(TRAIN_GRAPH_PATH))
    process_and_save(Path(VAL_GRAPH_PATH))
    process_and_save(Path(TEST_GRAPH_PATH))
