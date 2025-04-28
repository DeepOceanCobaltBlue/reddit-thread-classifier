# evaluate_saved_models.py

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from constants import DEVICE, BATCH_SIZE
from data_loader import load_graphs

# Paths
NODE2VEC_MODEL_PATH = "saved_models/node2vec_model.pt"
BERT_MODEL_PATH = "saved_models/bert_model.pt"
HYBRID_MODEL_PATH = "saved_models/hybrid_model_1.pt"

NODE2VEC_TEST_PATH = Path("datasets/test_graphs_n2v.pt")
BERT_HYBRID_TEST_PATH = Path("datasets/test_graphs.pt")  # BERT and Hybrid share

def evaluate(model_class, model_path, test_path, use_bert):
    print(f"\n🧪 Evaluating {model_class.__name__}...")

    graphs = load_graphs(test_path, use_bert=use_bert)
    test_loader = DataLoader(graphs, batch_size=BATCH_SIZE)

    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    from experiments.node2vec_only.model_node2vec import Node2VecGraphClassifier
    from experiments.bert_only.model_bert import BERTGraphClassifier
    from model import HybridGraphClassifier


    evaluate(Node2VecGraphClassifier, NODE2VEC_MODEL_PATH, NODE2VEC_TEST_PATH, use_bert=False)
    evaluate(BERTGraphClassifier, BERT_MODEL_PATH, BERT_HYBRID_TEST_PATH, use_bert=True)
    evaluate(HybridGraphClassifier, HYBRID_MODEL_PATH, BERT_HYBRID_TEST_PATH, use_bert=True)
