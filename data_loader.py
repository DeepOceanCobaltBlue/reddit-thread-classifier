"""
This module loads and prepares the graph datasets for training and evaluation.

It embeds node texts using BERT (or loads cached embeddings if available),
ensures that each graph has Node2Vec structural embeddings,
caches processed graphs for faster reuse, and returns
PyTorch Geometric DataLoaders for the train, validation, and test splits.
"""


import torch
from torch_geometric.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
from utils.bert_embedder import encode_texts_with_bert
from constants import (
    TRAIN_GRAPH_PATH,
    VAL_GRAPH_PATH,
    TEST_GRAPH_PATH,
    BERT_MODEL_NAME,
    BATCH_SIZE,
    DEVICE,
    NODE2VEC_DIM,
    MAX_TOKEN_LENGTH,
)

# 🔄 Load BERT model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()

# ✅ Batched BERT encoding function
def encode_bert(texts, batch_size=64):
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
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

def load_graphs(path: Path) -> list:
    embedded_path = Path(str(path).replace(".pt", "_embedded.pt"))

    # ✅ If already embedded, use it directly
    if embedded_path.exists():
        print(f"⚡ Using cached embedded graphs from: {embedded_path.name}")
        return torch.load(embedded_path)

    # 📦 Otherwise, load raw and embed
    print(f"📦 Loading raw graphs from: {path}")
    graphs = torch.load(path)

    print("🧠 Encoding BERT embeddings per graph...")
    for graph in tqdm(graphs):
        node_texts = graph.x

        # ✅ Only encode if node_texts are strings (not already embedded tensors)
        if isinstance(node_texts[0], str):
            graph.x = encode_texts_with_bert(node_texts)

        if not hasattr(graph, "struct_emb"):
            graph.struct_emb = torch.zeros((len(graph.x), NODE2VEC_DIM))

    torch.save(graphs, embedded_path)
    print(f"✅ Saved embedded version: {embedded_path.name}")
    return graphs

# ✅ Return dataloaders
def get_dataloaders():
    train_graphs = load_graphs(TRAIN_GRAPH_PATH)
    val_graphs = load_graphs(VAL_GRAPH_PATH)
    test_graphs = load_graphs(TEST_GRAPH_PATH)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader
