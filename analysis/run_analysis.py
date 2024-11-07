import sys
import os
import torch
import json
import numpy as np
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_base_path = os.path.join(script_dir, "./data/embeddings/")

term_pairs = []

models = ["llama", "pythia28"]
versions = ["policy", "reference"]

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

JSONEXT = ".json"
PTEXT = ".pt"


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
    run_terms_analysis() # this will require that you have a set of pairs of components you want to analyze




if __name__ == '__main__':
    run_kg_graph_analysis()