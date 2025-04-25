import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from constants import TEST_GRAPH_PATH, MODEL_SAVE_PATH, DEVICE, BATCH_SIZE
from data_loader import load_graphs
from model import HybridGraphClassifier

def evaluate():
    print("📦 Loading test graphs...")
    test_graphs = load_graphs(TEST_GRAPH_PATH)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    print("📥 Loading trained model...")
    model = HybridGraphClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(DEVICE)
            struct = batch.struct_emb.to(DEVICE)
            edge_index = batch.edge_index.to(DEVICE)
            batch_idx = batch.batch.to(DEVICE)
            y = batch.y.to(DEVICE)

            logits = model(batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # Evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print("\n🧪 Test Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate()
