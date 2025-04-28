"""
Cleans the embedded graph datasets for Node2Vec-only experiments.

- Loads existing *embedded* datasets (with Node2Vec struct embeddings only).
- Skips any graphs where edge_index has invalid indices.
- Saves cleaned datasets to *_n2v.pt files for safe Node2Vec-only training.

Run:
    python clean_graphs_for_node2vec.py
"""

import torch
from pathlib import Path
from tqdm import tqdm

from constants import (
    TRAIN_GRAPH_PATH,
    VAL_GRAPH_PATH,
    TEST_GRAPH_PATH,
)

def clean_graphs(input_path: Path, output_path: Path):
    print(f"\n📦 Loading: {input_path.name}")
    graphs = torch.load(input_path)

    cleaned_graphs = []
    skipped = 0

    for graph in tqdm(graphs, desc=f"Cleaning {input_path.name}"):
        num_nodes = graph.struct_emb.size(0)  # 🔥 Important fix: use struct_emb, not x
        if graph.edge_index.max() >= num_nodes:
            skipped += 1
            continue
        cleaned_graphs.append(graph)

    print(f"✅ Kept {len(cleaned_graphs)} graphs, Skipped {skipped} invalid graphs.")
    torch.save(cleaned_graphs, output_path)
    print(f"💾 Saved cleaned graphs to: {output_path.name}")

if __name__ == "__main__":
    clean_graphs(Path("datasets/train_graphs_embedded.pt"), Path("datasets/train_graphs_n2v.pt"))
    clean_graphs(Path("datasets/val_graphs_embedded.pt"), Path("datasets/val_graphs_n2v.pt"))
    clean_graphs(Path("datasets/test_graphs_embedded.pt"), Path("datasets/test_graphs_n2v.pt"))
