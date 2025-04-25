__SETUP OPTION 1__
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

__SETUP OPTION 2__
## **Alternative: Use the Setup Script**
Instead of manually running the setup steps, you can use the provided **setup_env.sh** script:

## in bash run the following commands
chmod +x setup_env.sh
./setup_env.sh

This will automatically:
    Create a virtual environment named venv310
    Upgrade pip
    Install the Python dependencies
    Install PyTorch Geometric modules
    Exit with error messages if any step fails

then activate venv: source venv310/bin/activate


Procedure to train a model:
0. Have datasets setup 
    a. /datasets
            /'subreddit 1'
                /comments
                    /scraped.jsonl
                /submissions
                    /scraped.jsonl
            /'subreddits 2'

Run these modules
1. preprocess_data.py
2. dataset_split.py 
3. run_node2vec_on_all.py	
4. train.py
5. evaluate.py 

Folder Structure:
/project-directory
    /datasets
    /logs
    /saved_models
    /scripts
    /constants.py
    /data_loader.py
    /dataset_split.py
    /embed_test_set.py
    /evaluate.py
    /model.py
    /preprocessing_data.py
    /run_node2vec_on_all.py
    /train.py
    /requirements.txt
    /setup_env.sh

