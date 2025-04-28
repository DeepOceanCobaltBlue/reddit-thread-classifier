#!/bin/bash

echo "🚿 Full Clean Node2Vec-Only Pipeline Starting..."

# 1. Clean previous graph files (except subreddit folders)
echo "🧹 Cleaning datasets folder..."
find datasets -type f ! -path "datasets/AskReddit/*" ! -path "datasets/science/*" ! -path "datasets/cooking/*" ! -path "datasets/showerthoughts/*" ! -path "datasets/politics/*" -delete

# 2. Preprocess subreddit directories into all_graphs.pt
echo "📚 Preprocessing subreddit data into graphs..."
python preprocessing_data.py

# 3. Split graphs into train/val/test BEFORE embedding
echo "✂️ Splitting graphs into train/val/test..."
python dataset_split.py

# 4. Run Node2Vec embedding
# Replace old broken Node2Vec run
echo "🧠 Embedding Node2Vec structure only..."
python node2vec_embed.py


# 5. Clean graphs for Node2Vec-only training
echo "🧹 Cleaning graphs for Node2Vec-only training..."
python clean_graphs_for_node2vec.py

# 6. Train Node2Vec-only model
echo "🏋️ Starting Node2Vec-only training..."
python experiments/node2vec_only/train_node2vec_only.py

# 7. Evaluate Node2Vec-only model
echo "🧪 Evaluating Node2Vec-only model..."
python experiments/node2vec_only/evaluate_node2vec_only.py

echo "✅ Full Node2Vec-only pipeline complete!"
