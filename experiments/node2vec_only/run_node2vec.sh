#!/bin/bash

# Run from project root: ~/SC/Final
echo "🧹 Cleaning old saved model if exists..."
rm -f saved_models/node2vec_model.pt

echo "📚 Starting training for Node2Vec-only model..."
python -m experiments.node2vec_only.train_node2vec_only

echo "🧪 Evaluating trained Node2Vec-only model on test set..."
python -m experiments.node2vec_only.evaluate_node2vec_only

echo "✅ Done! Node2Vec-only experiment complete."
