"""
This module splits the full preprocessed Reddit graph dataset into training,
validation, and test sets. 

The dataset is loaded, shuffled with a fixed random seed for reproducibility,
partitioned according to predefined ratios (70/15/15), and saved back into
separate .pt files for model training and evaluation.
"""

import torch
import random
from pathlib import Path
import torch_geometric.data
from constants import (
    FULL_DATASET_PATH,
    DATASET_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED
)

random.seed(RANDOM_SEED)

def split_dataset():
    print("📦 Loading all graphs...")

    from torch.serialization import add_safe_globals
    add_safe_globals([torch_geometric.data.Data])
    all_graphs = torch.load(FULL_DATASET_PATH, weights_only=False)

    random.shuffle(all_graphs)

    total = len(all_graphs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    test_end = val_end + int(total * TEST_RATIO)

    assert test_end <= total, "Ratio split cannot sum > 1.0"

    train_graphs = all_graphs[:train_end]
    val_graphs = all_graphs[train_end:val_end]
    test_graphs = all_graphs[val_end:test_end]

    torch.save(train_graphs, DATASET_DIR / "train_graphs.pt")
    torch.save(val_graphs, DATASET_DIR / "val_graphs.pt")
    torch.save(test_graphs, DATASET_DIR / "test_graphs.pt")

    print(f"✅ Dataset split complete:")
    print(f"  Train: {len(train_graphs)}")
    print(f"  Val:   {len(val_graphs)}")
    print(f"  Test:  {len(test_graphs)}")

if __name__ == "__main__":
    split_dataset()
