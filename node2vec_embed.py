"""
Embeds all graphs with Node2Vec structure only (no BERT).

Takes raw all_graphs.pt -> train/val/test, assigns dummy struct_emb if missing.
"""

import torch
from pathlib import Path
from tqdm import tqdm
from constants import NODE2VEC_DIM

# Paths
DATASET_DIR = Path("datasets")
TRAIN_PATH = DATASET_DIR / "train_graphs.pt"
VAL_PATH = DATASET_DIR / "val_graphs.pt"
TEST_PATH = DATASET_DIR / "test_graphs.pt"

def embed_struct_only(graphs):
    for graph in tqdm(graphs, desc="Embedding Node2Vec struct"):
        if not hasattr(graph, "struct_emb"):
            graph.struct_emb = torch.zeros((graph.x.size(0), NODE2VEC_DIM))
    return graphs

def process_and_save(input_path: Path):
    graphs = torch.load(input_path)
    graphs = embed_struct_only(graphs)
    output_path = input_path.with_name(input_path.stem + "_embedded.pt")
    torch.save(graphs, output_path)
    print(f"✅ Saved {output_path.name}")

if __name__ == "__main__":
    process_and_save(TRAIN_PATH)
    process_and_save(VAL_PATH)
    process_and_save(TEST_PATH)
