import os

# Define the base directory for the analysis folder
base_dir = "../analysis"

# Define the subdirectories to be created
folders = [
    "connected_components",
    "terms_analysis",
    "RAG_question_answering"
]

# Define the subfolders for each model and alignment type
models = ["llama_policy", "llama_reference", "pythia28_policy", "pythia28_reference"]

# Create the folder structure
for folder in folders:
    for model in models:
        path = os.path.join(base_dir, folder, model)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
