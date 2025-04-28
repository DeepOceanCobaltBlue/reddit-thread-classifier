"""
Model for BERT-only Reddit thread classification.
Uses BERT embeddings from node texts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from constants import (
    TEXT_EMBED_DIM, 
    GNN_HIDDEN_DIM, 
    GNN_NUM_LAYERS, 
    FUSION_HIDDEN_DIM, 
    NUM_CLASSES
)

class BERTGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(TEXT_EMBED_DIM, GNN_HIDDEN_DIM))
        for _ in range(GNN_NUM_LAYERS - 1):
            self.convs.append(GCNConv(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM))

        self.classifier = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out
