# full_embed_graphs.py
"""
Fully embeds graphs with both BERT and Node2Vec features.
"""

import torch
import random
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
from constants import (
    TRAIN_GRAPH_PATH,
    VAL_GRAPH_PATH,
    TEST_GRAPH_PATH,
    DEVICE,
    NODE2VEC_DIM,
    MAX_TOKEN_LENGTH,
    BERT_MODEL_NAME,
)
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import Node2Vec

# Load BERT
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()

# Node2Vec model parameters (same across graphs)
N2V_WALK_LENGTH = 20
N2V_CONTEXT_SIZE = 10
N2V_WALKS_PER_NODE = 10
N2V_P = 1
N2V_Q = 1

def embed_bert(texts, batch_size=64):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)
            outputs = bert_model(**encoded_input)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


def embed_struct(graph: Data) -> torch.Tensor:
    node2vec = Node2Vec(
        graph.edge_index, 
        embedding_dim=NODE2VEC_DIM, 
        walk_length=N2V_WALK_LENGTH,
        context_size=N2V_CONTEXT_SIZE,
        walks_per_node=N2V_WALKS_PER_NODE,
        p=N2V_P,
        q=N2V_Q,
        sparse=True
    ).to(DEVICE)

    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)
    
    node2vec.train()
    for _ in range(50):  # 50 mini-epochs
        optimizer.zero_grad()

        # ✅ Instead of all nodes, sample a random subset (batch)
        batch_size = min(128, graph.num_nodes)  # sample up to 128 nodes or all if small
        batch = torch.randint(0, graph.num_nodes, (batch_size,), device=DEVICE)

        pos_rw, neg_rw = node2vec.sample(batch)
        loss = node2vec.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()

    emb = node2vec.embedding.weight.detach().cpu()
    return emb


def process_and_save(input_path: Path, output_path: Path):
    graphs = torch.load(input_path)
    updated_graphs = []

    for graph in tqdm(graphs, desc=f"Embedding {input_path.name}"):
        # 1. BERT
        if isinstance(graph.x[0], str):
            graph.x = embed_bert(graph.x.tolist())

        # 2. Node2Vec
        graph.struct_emb = embed_struct(graph)

        updated_graphs.append(graph)

    torch.save(updated_graphs, output_path)
    print(f"✅ Saved fully embedded graphs to: {output_path.name}")

if __name__ == "__main__":
    process_and_save(TRAIN_GRAPH_PATH, Path("datasets/train_graphs_embedded.pt"))
    process_and_save(VAL_GRAPH_PATH, Path("datasets/val_graphs_embedded.pt"))
    process_and_save(TEST_GRAPH_PATH, Path("datasets/test_graphs_embedded.pt"))
