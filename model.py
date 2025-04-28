"""
This module defines the core model architecture for the Reddit thread classifier.

It contains:
- AttentionFusion: a learnable layer to combine BERT and Node2Vec embeddings.
- HybridGraphClassifier: the full model that fuses text + structure,
  applies a two-layer GCN, and predicts the subreddit class.

Expected inputs:
- Graph data with BERT node features and Node2Vec structural embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from constants import (
    BERT_DIM,
    NODE2VEC_DIM,
    FUSION_DIM,
    GNN_HIDDEN_DIM,
    NUM_CLASSES
)

class AttentionFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim1, out_dim)
        self.linear2 = nn.Linear(dim2, out_dim)
        self.attn = nn.Linear(out_dim, 1)

    def forward(self, x1, x2):
        h1 = self.linear1(x1)
        h2 = self.linear2(x2)

        # Learnable attention weights
        alpha1 = self.attn(h1)
        alpha2 = self.attn(h2)
        alphas = torch.cat([alpha1, alpha2], dim=1)
        weights = F.softmax(alphas, dim=1)

        # Weighted sum
        fused = weights[:, 0:1] * h1 + weights[:, 1:2] * h2
        return fused

class HybridGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Fuses BERT + Node2Vec embeddings
        self.fusion = AttentionFusion(BERT_DIM, NODE2VEC_DIM, FUSION_DIM)

        # GNN layers
        self.conv1 = GCNConv(FUSION_DIM, GNN_HIDDEN_DIM)
        self.conv2 = GCNConv(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM)

        # Graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(GNN_HIDDEN_DIM // 2, NUM_CLASSES)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 🛠 PROPER: Use the real Node2Vec embeddings
        struct_emb = data.struct_emb

        # Fuse BERT (x) and Node2Vec (struct_emb)
        x = self.fusion(x, struct_emb)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out
