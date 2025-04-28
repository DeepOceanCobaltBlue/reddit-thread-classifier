#!/bin/bash

# ==============================================
# BERT-only Experiment Runner
# ==============================================

echo "🚀 Activating virtual environment..."
source venv310/bin/activate

echo "🧹 Cleaning old saved model if exists..."
rm -f saved_models/bert_only_model.pt

echo "📚 Starting training for BERT-only model..."
python -m experiments.bert_only.train_bert

echo "🧪 Evaluating trained BERT-only model on test set..."
python -m experiments.bert_only.evaluate_bert

echo "✅ Done! BERT-only experiment complete."
