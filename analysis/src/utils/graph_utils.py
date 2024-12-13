import json
import torch
from torch_geometric.data import Data
import analysis_constants
import os

def load_kg_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error {e} in reading JSON file from {json_file_path}.")
        return None
    return data

def load_kg_data(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error {e} in reading JSON file: {json_file_path}.")
        return None
    return data

def prepare_knowledge_graph(knowledge_graph):
    entities = list(set([ent for ent in     ([r[0] for r in knowledge_graph[analysis_constants.RELATIONS_KEY]] + [r[2] for r in knowledge_graph[analysis_constants.RELATIONS_KEY]] )  ]))
    relations = list(set([rel[1] for rel in knowledge_graph[analysis_constants.RELATIONS_KEY]]))

    entity_to_idx = {ent: idx for idx, ent in enumerate(entities)}
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}

    return {
        analysis_constants.ENTITIES_KEY: entities,
        analysis_constants.RELATIONS_KEY: relations,
        analysis_constants.ENTITY_MAPPING_KEY : entity_to_idx,
        analysis_constants.RELATION_MAPPING_KEY: relation_to_idx,
        analysis_constants.TRIPLES_KEY: knowledge_graph[analysis_constants.JSON_RELATIONS_KEY] 
    }

def load_saved_graph_data(saved_graph_info_dir):
    graph_name = get_graph_name(saved_graph_info_dir)
    graph_data_path = get_graph_data_path(saved_graph_info_dir, graph_name)
    entity_embeddings_path = get_embeddings_path(saved_graph_info_dir, graph_name)
    relation_embeddings_path = get_relation_embeddings_path(saved_graph_info_dir, graph_name)  
    
    graph_data = load_kg_data(graph_data_path)
    entity_embeddings = torch.load(entity_embeddings_path).weight.detach().cpu().numpy() 
    relation_embeddings = torch.load(relation_embeddings_path).weight.detach().cpu().numpy() 

    return entity_embeddings, relation_embeddings, graph_data

def get_embeddings_path(save_folder_path, graph_name): 
    entity_embeddings_path = os.path.join(save_folder_path, f"{graph_name}_entity_embeddings.pt")
    return entity_embeddings_path

def get_relation_embeddings_path(save_folder_path, graph_name): 
    relation_embeddings_path = os.path.join(save_folder_path, f"{graph_name}_relation_embeddings.pt")
    return relation_embeddings_path

def get_graph_data_path(save_folder_path, graph_name): 
    graph_data_path = os.path.join(save_folder_path, f"{graph_name}_graph_data.json")
    return graph_data_path

def get_graph_name(graph_path):
    return os.path.basename(graph_path)
    

def get_plot_image_path(saved_graph_info_dir, plot_dir): 
    graph_name = get_graph_name(saved_graph_info_dir)
    plot_image_path = os.path.join(plot_dir, f"{graph_name}.png")
    return plot_image_path

def get_imputation_results_path(saved_graph_info_dir, output_dir): 
    graph_name = get_graph_name(saved_graph_info_dir)
    imputation_results_path = os.path.join(output_dir, f"{graph_name}_imputation_results.json")
    return imputation_results_path

def get_query_results_path(saved_graph_info_dir, output_dir): 
    graph_name = get_graph_name(saved_graph_info_dir)
    query_results_path = os.path.join(output_dir, f"{graph_name}_query_results.json")
    return query_results_path

def get_index_name(saved_graph_info_dir):
    graph_name = get_graph_name(saved_graph_info_dir)
    index_name = f"{graph_name}index"
    #remove special characters
    index_name = ''.join(c for c in index_name if c.isalnum())
    # should be less than 45 characters
    index_name = index_name[:45]
    return index_name

def get_save_folder_path(graph_path, output_path, _): 
    graph_name = get_graph_name(graph_path)
    save_folder_path = os.path.join(output_path, graph_name)
    return save_folder_path


def get_cluster_assignments_save_path(saved_graph_info_dir, output_dir): 
    graph_name = get_graph_name(saved_graph_info_dir)
    cluster_assignments_save_path = os.path.join(output_dir, f"{graph_name}_cluster_assignments.json")
    return cluster_assignments_save_path



def get_toxicity_results_save_path(saved_graph_info_dir, output_dir): 
    graph_name = get_graph_name(saved_graph_info_dir)
    toxicity_results_save_path = os.path.join(output_dir, f"{graph_name}_toxicity_results.json")
    return toxicity_results_save_path


def get_new_triples_analysis_save_path(saved_new_triples_path, output_dir):
    graph_name_with_extension = os.path.basename(saved_new_triples_path)
    graph_name = '.'.join(graph_name_with_extension.split('.')[:-1])
    new_triples_analysis_save_path = os.path.join(output_dir, f"{graph_name}_new_triples_analysis.json")
    return new_triples_analysis_save_path

def get_cluster_states_results_save_path(saved_new_triples_path, output_dir):    
    graph_name_with_extension = os.path.basename(saved_new_triples_path)
    graph_name = '.'.join(graph_name_with_extension.split('.')[:-1])
    cluster_states_results_save_path = os.path.join(output_dir, f"{graph_name}_cluster_states_results.json")
    return cluster_states_results_save_path