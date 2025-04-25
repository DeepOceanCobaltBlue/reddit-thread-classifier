"""
This module processes scraped Reddit threads into graph datasets
for model training.

Main points:
- Runs preprocessing across all subreddits listed in SUBREDDITS.
- Expects each subreddit to exist in datasets/ with scraped.jsonl files.
- Filters out posts and comments marked as '[removed]' or '[deleted]'.
- For each submission + its comments, builds a graph using utils/graph_tools.py > thread_to_graph().
- Saves the full processed dataset to datasets/all_graphs.pt.

Submission processing:
- Submissions must have a non-removed title or selftext.
- Submissions must have at least MIN_COMMENTS.
- Only comments with real text (not removed/deleted) are used.
"""

import json
from pathlib import Path
from typing import List, Dict
import torch
from torch_geometric.data import Data
from utils.graph_tools import thread_to_graph
from constants import ( 
    MIN_COMMENTS, 
    MAX_COMMENTS, 
    DATASET_DIR, 
    SUBREDDITS 
)

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

    # Filter for good submissions
    sub_map = {
        s["id"]: s for s in submissions
        if (
            (s.get("title") or s.get("selftext")) and
            (s.get("title", "").lower() not in ("[removed]", "[deleted]")) and
            (s.get("selftext", "").lower() not in ("[removed]", "[deleted]")) and
            s.get("num_comments", 0) >= MIN_COMMENTS
        )
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
