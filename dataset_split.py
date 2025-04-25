import torch
import random
from pathlib import Path
import torch_geometric.data

# Load the dataset
DATASET_DIR = Path("datasets")
INPUT_FILE = DATASET_DIR / "all_graphs.pt"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def split_dataset():
    print("📦 Loading all graphs...")

    from torch.serialization import add_safe_globals
    add_safe_globals([torch_geometric.data.Data])
    all_graphs = torch.load(INPUT_FILE, weights_only=False)

    random.shuffle(all_graphs)

    total = len(all_graphs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_graphs = all_graphs[:train_end]
    val_graphs = all_graphs[train_end:val_end]
    test_graphs = all_graphs[val_end:]

    torch.save(train_graphs, DATASET_DIR / "train_graphs.pt")
    torch.save(val_graphs, DATASET_DIR / "val_graphs.pt")
    torch.save(test_graphs, DATASET_DIR / "test_graphs.pt")

    print(f"✅ Dataset split complete:")
    print(f"  Train: {len(train_graphs)}")
    print(f"  Val:   {len(val_graphs)}")
    print(f"  Test:  {len(test_graphs)}")

if __name__ == "__main__":
    split_dataset()
