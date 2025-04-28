"""
Training script for Node2Vec-only model.
"""

import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader
import os
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from constants import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LOG_DIR
from experiments.node2vec_only.model_node2vec import Node2VecGraphClassifier
from data_loader import load_graphs

# Paths for N2V-only
TRAIN_GRAPH_PATH = Path("datasets/train_graphs_n2v.pt")
VAL_GRAPH_PATH = Path("datasets/val_graphs_n2v.pt")
TEST_GRAPH_PATH = Path("datasets/test_graphs_n2v.pt")

MODEL_SAVE_PATH = "saved_models/node2vec_model.pt"
LOG_FILE = LOG_DIR / "train_log_node2vec.csv"

def get_n2v_dataloaders():
    train_graphs = load_graphs(TRAIN_GRAPH_PATH, use_bert=False)
    val_graphs = load_graphs(VAL_GRAPH_PATH, use_bert=False)
    test_graphs = load_graphs(TEST_GRAPH_PATH, use_bert=False)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    return train_loader, val_loader, test_loader

def evaluate(model, dataloader):
    model.eval()
    correct, total, losses = 0, 0, []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            losses.append(loss.item())

            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)

    acc = correct / total if total > 0 else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss, acc

def train():
    train_loader, val_loader, _ = get_n2v_dataloaders()
    model = Node2VecGraphClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        log.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved with val acc: {val_acc:.4f}")

    os.makedirs(LOG_DIR, exist_ok=True)
    pd.DataFrame(log).to_csv(LOG_FILE, index=False)
    print(f"📈 Training log saved to {LOG_FILE}")

if __name__ == "__main__":
    train()
