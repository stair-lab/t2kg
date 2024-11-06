import json
import torch
from torch_geometric.data import Data


def load_kg_data(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    entities = data["entities"]
    relations = data["relations"]

    # Map each entity to a unique ID
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}

    # Map each unique relation type to an ID
    relation_types = list(set(relation[1] for relation in relations))
    relation_to_id = {relation: idx for idx, relation in enumerate(relation_types)}

    # Create edge_index and edge_type lists
    edge_index = []
    edge_type = []

    for head, relation, tail in relations:
        head_idx = entity_to_idx[head]
        tail_idx = entity_to_idx[tail]
        relation_idx = relation_to_id[relation]

        edge_index.append([head_idx, tail_idx])
        edge_type.append(relation_idx)

    # Convert lists to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Shape [2, num_edges]
    edge_type = torch.tensor(edge_type, dtype=torch.long)  # Shape [num_edges]

    # Number of nodes is the number of unique entities
    num_nodes = len(entities)

    # Create the PyG data object
    pyg_data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_type=edge_type)

    return pyg_data

# Example usage:
# data = load_kg_data('path_to_your_file.json')
# print(data)
