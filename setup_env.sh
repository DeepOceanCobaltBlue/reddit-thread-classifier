#!/bin/bash

# ==============================================
# Environment Setup Script for Reddit Classifier
# Python 3.10 + CPU-only PyTorch + PyTorch Geometric
# use: chmod +x setup_env.sh
# then: ./setup_env.sh
# ==============================================

echo "📦 Creating Python 3.10 virtual environment (venv310)..."
python3.10 -m venv venv310 || { echo "❌ Failed to create venv. Is Python 3.10 installed?"; exit 1; }
source venv310/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📄 Installing base dependencies from requirements.txt..."
pip install -r requirements.txt || { echo "❌ Failed to install from requirements.txt"; exit 1; }

echo "📦 Installing PyTorch Geometric and structural components..."
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html || {
    echo "❌ Failed to install PyG modules"
    exit 1
}

echo "✅ Setup complete!"
echo "To activate your environment later, run: source venv310/bin/activate"