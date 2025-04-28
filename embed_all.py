from data_loader import embed_node_texts
from constants import RAW_TRAIN_GRAPH_PATH, RAW_VAL_GRAPH_PATH, RAW_TEST_GRAPH_PATH
import torch
from pathlib import Path

CHUNK_SIZE = 100  # graphs at a time

def embed_and_save_in_chunks(raw_path, out_path):
    print(f"📦 Embedding graphs from: {raw_path.name}")
    graphs = torch.load(raw_path)

    embedded_graphs = []
    for i in range(0, len(graphs), CHUNK_SIZE):
        chunk = graphs[i:i+CHUNK_SIZE]
        print(f"🧠 Embedding graphs {i} to {i+len(chunk)-1}...")

        for graph in chunk:
            if isinstance(graph.x[0], str):
                graph.x = embed_node_texts(graph.x)  
            if not hasattr(graph, "struct_emb"):
                graph.struct_emb = torch.zeros((len(graph.x), 64))
        embedded_graphs.extend(chunk)

        torch.cuda.empty_cache()  # GPU cleanup if needed

    print(f"💾 Saving embedded graphs to: {out_path.name}")
    torch.save(embedded_graphs, out_path)

if __name__ == "__main__":
    embed_and_save_in_chunks(RAW_TRAIN_GRAPH_PATH, Path("datasets/train_graphs_embedded.pt"))
    embed_and_save_in_chunks(RAW_VAL_GRAPH_PATH,   Path("datasets/val_graphs_embedded.pt"))
    embed_and_save_in_chunks(RAW_TEST_GRAPH_PATH,  Path("datasets/test_graphs_embedded.pt"))
