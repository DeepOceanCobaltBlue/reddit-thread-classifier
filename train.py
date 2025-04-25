import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader
import pandas as pd
import os

from data_loader import get_dataloaders
from model import HybridGraphClassifier
from constants import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_SAVE_PATH, LOG_FILE
)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = cross_entropy(out, batch.y)
            losses.append(loss.item())
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)

    acc = correct / total
    avg_loss = sum(losses) / len(losses)
    return avg_loss, acc

def train():
    train_loader, val_loader, _ = get_dataloaders()

    model = HybridGraphClassifier().to(DEVICE)
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

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

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

    # Save log to CSV
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    pd.DataFrame(log).to_csv(LOG_FILE, index=False)
    print(f"📈 Training log saved to {LOG_FILE}")

if __name__ == "__main__":
    train()
