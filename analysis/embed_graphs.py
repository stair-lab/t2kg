# import sys
import os
# from utils import graph_utils
# from utils import RGCN
# import torch

# from utils import train_and_test_utils

# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
kg_path = os.path.join(script_dir, "../extracted_graphs/test_small.json")

embeddings_path = os.path.join(script_dir, "./embeddings/")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# args = {
#       'device': device,
#       'num_layers': 3,
#       'hidden_dim': 256,
#       'input_dim': 16, 
#       'output_dim': 16, 
#       'dropout': 0.5,
#       'lr': 0.01,
#       'epochs': 100,
#   }


# def run_embed_graph(kg_path, save_path): 

#     data = graph_utils.load_kg_data(kg_path)
#     num_relations = len(torch.unique(data.edge_type))
#     # load data 
#     model = RGCN(args['input_dim'], args['hidden_dim'], args['output_dim'], num_relations)
#     # Manually create splits
#     num_nodes = data.num_nodes
#     train_idx = torch.randperm(num_nodes)[:int(0.6 * num_nodes)]
#     val_idx = torch.randperm(num_nodes)[int(0.6 * num_nodes):int(0.8 * num_nodes)]
#     test_idx = torch.randperm(num_nodes)[int(0.8 * num_nodes):]

#     # Add splits as attributes
#     data.train_idx = train_idx
#     data.val_idx = val_idx
#     data.test_idx = test_idx

#     train_and_test_utils.train_and_evaluate(model, data, 
#                                             train_idx, split_idx, 
#                                             evaluator, args, 
#                                             save_model_results=False, 
#                                             checkpoint_dir="checkpoints"):


import json
import torch
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory  # Correct the import statement
import pandas as pd

def run_pykeen_embed_graph(kg, save_path): 
    # Load the JSON data
    with open(kg, 'r') as f:
        data = json.load(f)

    # Create mappings for entities and relations
    entities = data["entities"]
    relations = data["relations"]

    entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
    relation_to_id = {relation[1]: idx for idx, relation in enumerate(relations)}

    # Convert relations into (head, relation, tail) triples
    triples = []
    for relation in relations:
        head, rel, tail = relation
        # head_id = entity_to_id[head]
        # tail_id = entity_to_id[tail]
        # rel_id = relation_to_id[rel]
        triples.append((head, rel, tail))

    # Step 3: Create the TriplesFactory from the labeled triples
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    print(df)
    triples_factory = TriplesFactory.from_labeled_triples(
    triples=df[['head', 'relation', 'tail']].values,
)

    

    training = triples_factory
    validation = triples_factory
    testing = triples_factory

    # Train the model (TransE in this case)
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model='TransE',
        training_loop='slcwa',
        epochs=100,
        negative_sampler="basic",
        random_seed=42,
    )

    # Access the trained model
    model = result.model

    # View entity and relation embeddings
    entity_embeddings = model.entity_representations[0]().cpu().detach()
    
    relation_embeddings = model.relation_representations[0]().cpu().detach()

    # Example: Access embedding of the entity "U-Haul"
    entity_id = entity_to_id["U-Haul"]
    entity_embedding = entity_embeddings[entity_id]

    print("Entity Embedding for 'U-Haul':", entity_embedding)



if __name__ == '__main__':
    run_pykeen_embed_graph(kg_path, embeddings_path)