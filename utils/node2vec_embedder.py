import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

def compute_node2vec_embeddings(graphs, embedding_dim=64, walk_length=10, context_size=5, walks_per_node=10, num_negative_samples=1, p=1, q=1, epochs=20):
    for graph in tqdm(graphs, desc="Computing Node2Vec embeddings"):
        num_nodes = graph.num_nodes
        node2vec = Node2Vec(
            edge_index=graph.edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=True
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node2vec = node2vec.to(device)
        loader = node2vec.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

        node2vec.train()
        for _ in range(epochs):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        node2vec.eval()
        with torch.no_grad():
            emb = node2vec.embedding.weight.detach().cpu()
            graph.struct_emb = emb

    return graphs

if __name__ == "__main__":
    from pathlib import Path
    import torch

    path = Path("datasets/train_graphs.pt")
    graphs = torch.load(path)
    graphs = compute_node2vec_embeddings(graphs)
    torch.save(graphs, path)  # overwrite with structural embeddings
    print("✅ Node2Vec embeddings added.")