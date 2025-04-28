"""
Evaluation script for Node2Vec-only model.
"""

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from constants import DEVICE, BATCH_SIZE
from experiments.node2vec_only.model_node2vec import Node2VecGraphClassifier

# Override path for cleaned N2V graphs
TEST_GRAPH_PATH = Path("datasets/test_graphs_n2v.pt")
MODEL_SAVE_PATH = "saved_models/node2vec_model.pt"

from data_loader import load_graphs

def evaluate():
    print("📦 Loading test graphs...")
    test_graphs = load_graphs(TEST_GRAPH_PATH, use_bert=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    print("📥 Loading trained model...")
    model = Node2VecGraphClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    print("\n🧪 Test Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate()
