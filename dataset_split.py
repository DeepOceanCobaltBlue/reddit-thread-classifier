"""
Splits all_graphs.pt into train/val/test splits.

This script loads the full dataset (all_graphs.pt) and splits it into three sets:
- train_graphs.pt
- val_graphs.pt
- test_graphs.pt

It uses constant ratios defined in constants.py.
"""

import torch
import random
from pathlib import Path
from constants import DATASET_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED

# Set reproducibility
random.seed(RANDOM_SEED)

# Paths
INPUT_FILE = DATASET_DIR / "all_graphs.pt"
TRAIN_FILE = DATASET_DIR / "train_graphs.pt"
VAL_FILE = DATASET_DIR / "val_graphs.pt"
TEST_FILE = DATASET_DIR / "test_graphs.pt"

def split_dataset():
    print("📦 Loading all graphs...")
    graphs = torch.load(INPUT_FILE)
    random.shuffle(graphs)

    n = len(graphs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]

    print(f"✂️ Splitting: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs.")

    torch.save(train_graphs, TRAIN_FILE)
    torch.save(val_graphs, VAL_FILE)
    torch.save(test_graphs, TEST_FILE)
    print("✅ Dataset splits saved!")

if __name__ == "__main__":
    split_dataset()
