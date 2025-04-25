import json
from pathlib import Path
from typing import List, Dict
import torch
from torch_geometric.data import Data
from utils.graph_tools import thread_to_graph

# ======= Constants =======
MIN_COMMENTS = 5
MAX_COMMENTS = 100
DATASET_DIR = Path("datasets")
SUBREDDITS = ["AskReddit", "science", "politics", "cooking", "showerthoughts"]

# Build label mapping once
label_map = {name: i for i, name in enumerate(SUBREDDITS)}

# ======= Load JSONL Helper =======
def load_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items

# ======= Preprocess One Subreddit =======
def preprocess_subreddit(subreddit: str) -> List[Data]:
    print(f"Processing r/{subreddit}...")
    base_path = DATASET_DIR / subreddit
    sub_path = base_path / "submissions" / "scraped.jsonl"
    com_path = base_path / "comments" / "scraped.jsonl"

    submissions = load_jsonl(sub_path)
    comments = load_jsonl(com_path)

    # Map submission_id to submission
    sub_map = {
        s["id"]: s for s in submissions
        if (s.get("title") or s.get("selftext")) and s.get("num_comments", 0) >= MIN_COMMENTS
    }

    # Group comments by thread ID
    thread_comments: Dict[str, List[dict]] = {sid: [] for sid in sub_map}
    for comment in comments:
        sid = comment.get("link_id", "").replace("t3_", "")
        if sid in thread_comments and comment.get("body") and comment["body"].lower() not in ("[removed]", "[deleted]"):
            thread_comments[sid].append(comment)

    dataset = []

    for sid, sub in sub_map.items():
        thread = thread_comments.get(sid, [])
        if not (MIN_COMMENTS <= len(thread) <= MAX_COMMENTS):
            continue

        # Build node list for thread_to_graph
        nodes = []

        # Submission node
        sub_text = f"{sub.get('title', '')} {sub.get('selftext', '')}".strip()
        nodes.append({
            "id": sid,
            "text": sub_text,
            "parent_id": None,
            "subreddit": subreddit
        })

        # Comment nodes
        for comment in thread:
            nodes.append({
                "id": comment["id"],
                "text": comment["body"],
                "parent_id": comment.get("parent_id"),
                "subreddit": subreddit
            })

        # Build graph
        try:
            graph = thread_to_graph(nodes, label_map)
            dataset.append(graph)
        except Exception as e:
            print(f"⚠️ Skipping thread {sid} due to error: {e}")

    print(f"→ {len(dataset)} valid graphs for r/{subreddit}")
    return dataset

# ======= Full Dataset Build =======
def preprocess_all():
    all_graphs = []
    for sub in SUBREDDITS:
        all_graphs.extend(preprocess_subreddit(sub))

    torch.save(all_graphs, DATASET_DIR / "all_graphs.pt")
    print("✅ Saved all preprocessed graphs to all_graphs.pt")

# ======= Main =======
if __name__ == "__main__":
    preprocess_all()
