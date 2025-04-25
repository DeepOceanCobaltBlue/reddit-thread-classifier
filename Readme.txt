# Step 1: Create a new virtual environment using Python 3.10
python3.10 -m venv venv310
source venv310/bin/activate

# Step 2: Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Install PyTorch Geometric + structural extensions
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Optional: Fix NumPy version if compatibility warnings occur
pip install numpy==1.24.4
