import torch
from pathlib import Path

# ======== Preprocessing Config ========
MIN_COMMENTS = 5
MAX_COMMENTS = 100
DATASET_DIR = Path("datasets/")
SUBREDDITS = ["AskReddit", "science", "politics", "cooking", "showerthoughts"]

# ======== Model Architecture ========
BERT_DIM = 768
NODE2VEC_DIM = 64  
FUSION_DIM = 768  
GNN_HIDDEN_DIM  = 256
NUM_CLASSES = 5  # UPDATE CLASS COUNT WHEN NECESSARY

# ======== General Training Config ========
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3
SEED = 42

# ======== Paths ========
DATASET_DIR = Path("datasets/")
TRAIN_GRAPH_PATH = DATASET_DIR / "train_graphs_embedded.pt"
VAL_GRAPH_PATH   = DATASET_DIR / "val_graphs_embedded.pt"
TEST_GRAPH_PATH  = DATASET_DIR / "test_graphs_embedded.pt"

MODEL_SAVE_PATH = Path("saved_models/hybrid_model.pt")
LOG_DIR = Path("logs/")
LOG_FILE = LOG_DIR / "train_log.csv"

# ======== Text Embedding ========
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 128
TEXT_EMBED_DIM = 768  # DistilBERT CLS token size

# ======== Structural Embedding (Node2Vec) ========
NODE2VEC_DIM = 64
NODE2VEC_WALK_LENGTH = 10
NODE2VEC_CONTEXT_SIZE = 5
NODE2VEC_NUM_WALKS = 10
NODE2VEC_P = 1.0
NODE2VEC_Q = 1.0

# ======== GNN Layer Config ========
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 2

# ======== Fusion Layer ========
FUSION_HIDDEN_DIM = 128

# ======== Hardware ========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Dataset Split Config ========
FULL_DATASET_PATH = DATASET_DIR / "all_graphs.pt"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

RAW_TRAIN_GRAPH_PATH = DATASET_DIR / "train_graphs.pt"
RAW_VAL_GRAPH_PATH = DATASET_DIR / "val_graphs.pt"
RAW_TEST_GRAPH_PATH = DATASET_DIR / "test_graphs.pt"

