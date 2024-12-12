import os
import json
import torch
import analysis_constants 
import argparse
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
import embedding_models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    from cuml.cluster import KMeans as GPUKMeans
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

NUMBER_OF_CLUSTERS = 200 # some eye-balling from the aggregate stats in the paper
EXPECTED_CLUSTER_SIZE = 5

def save_ouput(save_folder_path, entity_embeddings, relation_embeddings, graph_data, graph_name): 
    print(f"Saving output to {save_folder_path}")
    os.makedirs(save_folder_path, exist_ok=True)

    entity_embeddings_path = graph_utils.get_embeddings_path(save_folder_path, graph_name)
    relation_embeddings_path = graph_utils.get_relation_embeddings_path(save_folder_path, graph_name)

    print(f"Saving embeddings to {entity_embeddings_path}")
    torch.save(entity_embeddings, entity_embeddings_path)
    torch.save(relation_embeddings, relation_embeddings_path)

    graph_data_path = graph_utils.get_graph_data_path(save_folder_path, graph_name)

    print(f"Saving graph data to {graph_data_path}")
    with open(graph_data_path, 'w') as f:
        json.dump(graph_data, f, indent=4)

def perform_clustering(entity_embeddings, num_clusters, use_gpu=False):
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(entity_embeddings)

    if use_gpu and CUDA_AVAILABLE:
        print("Using GPU for clustering")
        clustering_model = GPUKMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    else:
        print("Using CPU for clustering")
        clustering_model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42, n_init=10)
    
    cluster_labels = clustering_model.fit_predict(scaled_embeddings)
    return cluster_labels

def get_number_of_clusters(graph_data):
    # num_entities = len(graph_data[analysis_constants.ENTITIES_KEY])
    # return num_entities // expected_cluster_size
    return NUMBER_OF_CLUSTERS


def embed_graph(graph_path, ouput_path, model_name): 
    
    kg = graph_utils.load_kg_from_json(graph_path)
    graph_data = graph_utils.prepare_knowledge_graph(kg)

    print(f"Training knowledge graph embeddings using {model_name}")
    model = embedding_models.train_embeddings(graph_data, model_name)
    print(f"Finished training knowledge graph embeddings using {model_name}")


    use_gpu = CUDA_AVAILABLE

    # Extract entity and relation embeddings
    entity_embeddings = model.entity_embeddings
    relation_embeddings = model.relation_embeddings

    number_of_clusters = get_number_of_clusters(graph_data)

    np_embeddings = entity_embeddings.weight.detach().cpu().numpy()

    print(f"Clustering entities into {number_of_clusters} clusters")
    cluster_labels = perform_clustering(np_embeddings, num_clusters=number_of_clusters, use_gpu=use_gpu)
    graph_data[analysis_constants.ENTITY_CLUSTERS_KEY] = cluster_labels.tolist()


    graph_name = graph_utils.get_graph_name(graph_path)

    # save entity and relation embeddings as well as graph data 
    save_folder_path = graph_utils.get_save_folder_path(graph_path, ouput_path, model_name)
    save_ouput(save_folder_path, entity_embeddings, relation_embeddings, graph_data, graph_name)


def embed_folder(folder_path, output_path, model):
    """Embed all graphs in a folder and save them to the output path using the specified model."""
    print(f"Embedding all graphs in folder: {folder_path} -> Output Path: {output_path} using model: {model}")
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            embed_graph(file_path, output_path, model)

def embed_selected_graphs(graphs_list, output_path, model):
    """Embed a list of selected graphs and save them to the output path using the specified model."""
    print(f"Embedding selected graphs: {', '.join(graphs_list)} -> Output Path: {output_path} using model: {model}")
    for file_path in graphs_list:
        embed_graph(file_path, output_path, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed graphs based on input arguments.")

    # Input arguments

    parser.add_argument(
        "-d", "--directory", type=str, 
        default=analysis_constants.KNOWLEDGE_BASE_PATH, 
        help="Path to a folder containing multiple graph files."
    )

    # Output argument with default value
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default=analysis_constants.EMBEDDINGS_BASE_PATH, 
        help="Path to save the embedded graphs. Default is './output'."
    )

    # Embedding model argument
    parser.add_argument(
        "-m", 
        "--model", 
        type=str, 
        choices=["TransR"], 
        default="TransR", 
        help= f"Choose the embedding model to use. Default is TransR."
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Output path: {args.output}")


    embed_folder(args.directory, args.output, args.model)
    