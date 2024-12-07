import sys
import os
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_base_path = os.path.join(script_dir, "./data/embeddings/")
plots_base_path = os.path.join(script_dir, "./data/plots/")

term_pairs = []


def load_entity_embedding(entity_pair, embeddings_filename, entity_to_id_filename):

    print(f"Loading embeddings and entiry from {embeddings_filename, entity_to_id_filename}")
    # Load entity embeddings
    entity_embeddings = torch.load(embeddings_filename)


    print("The dimensions of the embeddings is: ", entity_embeddings.shape)
    # Load entity to ID mapping
    with open(entity_to_id_filename, 'r') as f:
        entity_to_id = json.load(f)

    # Retrieve the ID for the entity name
    entity_id0 = entity_to_id.get(entity_pair[0])
    entity_id1 = entity_to_id.get(entity_pair[1])


    # Check if entity exists in the mapping
    if entity_id0 is None or entity_id1 is None:
        print(f"Could not find one of {entity_pair} in the knowledge graph.")
        return None, None
    print(f"The id for the two entities are {(entity_id0, entity_id1)}")
    return entity_embeddings[entity_id0], entity_embeddings[entity_id1]


def load_entity_embeddings_and_ids(embeddings_filename, entity_to_id_filename): 
    entity_embeddings = torch.load(embeddings_filename)


    with open(entity_to_id_filename, 'r') as f:
        entity_to_id = json.load(f)

    return entity_embeddings, entity_to_id

JSONEXT = ".json"
PTEXT = ".pt"


def visualize_embeddings():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    kg_name_without_ext = f"harmless_base_rejected_test_50"
    entity_to_id_filename = os.path.join(embeddings_base_path, kg_name_without_ext + "_entity_to_id.json")
    embeddings_filename = os.path.join(embeddings_base_path, kg_name_without_ext + "_entity_embeddings.pt")
    
    embeddings, entity_to_id = load_entity_embeddings_and_ids(embeddings_filename, entity_to_id_filename)
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    # Convert embeddings to numpy and perform PCA to reduce dimensionality to 2D
    embeddings_array = np.array([emb.numpy() for emb in embeddings])
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    # Plot each point with an annotation for the entity name
    for i, (x, y) in enumerate(reduced_embeddings):
        ax.scatter(x, y, color='blue', s=20)  # Plot the point
        entity_name = id_to_entity.get(i, f"Entity {i}")

    ax.set_title(f"Embedding Visualization for Model: {kg_name_without_ext}")
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")

    # Save the plot
    plot_filename = f"{kg_name_without_ext}_embeddings.png"
    plt_path = os.path.join(plots_base_path, plot_filename)
    plt.savefig(plt_path)
    plt.close()
    print(f"Saved plot at {plt_path}")


def cosine_similarity(v1, v2):
    # Add batch dimension if needed
    if v1.dim() == 1:
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)
    return F.cosine_similarity(v1, v2, dim=1)

def run_terms_analysis(): 
    for tp in term_pairs: 
        for m in models: 
            for v in versions: 
                kg_name_without_ext = f"{m}_{v}"
        

                # prefix = os.path.splitext(os.path.basename(kg_path))[0]
                entity_to_id_filename = os.path.join(embeddings_base_path, kg_name_without_ext + "_entity_to_id" + JSONEXT)
                embeddings_filename = os.path.join(embeddings_base_path, kg_name_without_ext + "_entity_embeddings" + PTEXT)
                # Run function
                
                e1, e2 = load_entity_embedding(tp, embeddings_filename, entity_to_id_filename)
                if e1 is None or e2 is None: 
                    continue
                print(f"The similarity of {(tp)} for {m}, {v} is {cosine_similarity(e1, e2)}")

def run_kg_graph_analysis():
    #un_terms_analysis() # this will require that you have a set of pairs of components you want to analyze
    visualize_embeddings()




if __name__ == '__main__':
    ""