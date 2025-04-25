from data_loader import load_graphs
from pathlib import Path

# This is the *raw* test graph file (already exists)
raw_test_path = Path("datasets/test_graphs.pt")

# This call will:
# 1. Load raw test graphs
# 2. Run BERT embedding (using encode_texts_with_bert)
# 3. Attach struct_emb (if missing)
# 4. Save to test_graphs_embedded.pt
print("📦 Embedding test graphs...")
load_graphs(raw_test_path)
