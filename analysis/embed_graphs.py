import os
import json
import torch
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import pandas as pd

def run_pykeen_embed_graph(kg, save_path): 
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Load the JSON data
    with open(kg, 'r') as f:
        data = json.load(f)

    # Create mappings for entities and relations
    entities = data["entities"]
    relations = data["relations"]
    

    rel_list = [rel[1] for rel in relations]
    unique_res_list = list(set(rel_list))


    entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
    relation_to_id = {relation: idx for idx, relation in enumerate(unique_res_list)}

    # print(entity_to_id)
    # print(relation_to_id)

    # Convert relations into (head, relation, tail) triples
    triples = [(head, rel, tail) for head, rel, tail in relations]

    # Step 3: Create the TriplesFactory from the labeled triples
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    triples_factory = TriplesFactory.from_labeled_triples(df[['head', 'relation', 'tail']].values)

    # Training, validation, and testing
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
        epochs=200,    # recommended in some o
        negative_sampler="basic",
        random_seed=42,
    )

    # Access the trained model
    model = result.model

    # Extract entity and relation embeddings
    entity_embeddings = model.entity_representations[0]().cpu().detach()
    relation_embeddings = model.relation_representations[0]().cpu().detach()

    # Define prefix
    prefix = prefix = os.path.splitext(os.path.basename(kg))[0]

    # Save embeddings with prefix
    torch.save(entity_embeddings, os.path.join(save_path, f'{prefix}_entity_embeddings.pt'))
    torch.save(relation_embeddings, os.path.join(save_path, f'{prefix}_relation_embeddings.pt'))

    # Save mappings with prefix
    with open(os.path.join(save_path, f'{prefix}_entity_to_id.json'), 'w') as f:
        json.dump(entity_to_id, f)
    with open(os.path.join(save_path, f'{prefix}_relation_to_id.json'), 'w') as f:
        json.dump(relation_to_id, f)

    print(f"Embeddings and mappings saved in: {save_path}")

if __name__ == "__main__": 
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kg_path = os.path.join(script_dir, "../extracted_graphs/test_small.json")
    embeddings_path = os.path.join(script_dir, "./data/embeddings/")

    models = ["llama", "pythia28"]
    versions = ["policy", "reference"]

    kg_base_path = os.path.join(script_dir, "../extracted_graphs/")
    embeddings_save_base_path = embeddings_path = os.path.join(script_dir, "./data/embeddings/")

    for m in models: 
        for v in versions: 
            kg_name = f"{m}_{v}.json"
            kg_path = os.path.join(kg_base_path, kg_name)
            # Run function
            run_pykeen_embed_graph(kg_path, embeddings_save_base_path)
