import argparse
import os
import numpy as np
import analysis_constants
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch

try:
    from cuml.cluster import KMeans as GPUKMeans
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

def load_embeddings(saved_graph_info_dir):
    entity_embeddings, _, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)
    return entity_embeddings, graph_data

def perform_clustering(embeddings, num_clusters, use_gpu):
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    if use_gpu and CUDA_AVAILABLE:
        print("Using GPU for clustering")
        clustering_model = GPUKMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    else:
        print("Using CPU for clustering")
        clustering_model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42, n_init=10)
    
    cluster_labels = clustering_model.fit_predict(scaled_embeddings)
    return cluster_labels

def save_cluster_assignments(output_dir, cluster_labels, graph_data):
    entity_to_id_map = graph_data[analysis_constants.ENTITY_MAPPING_KEY]
    id_to_entity_map = {v: k for k, v in entity_to_id_map.items()}
    
    cluster_assignments = {id_to_entity_map[idx]: cluster for idx, cluster in enumerate(cluster_labels)}
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cluster_assignments.json")
    
    with open(output_path, "w") as f:
        import json
        json.dump(cluster_assignments, f, indent=4)
    
    print(f"Cluster assignments saved to {output_path}")

def cluster_embeddings(saved_graph_info_dir, output_dir, num_clusters, use_gpu):
    embeddings, graph_data = load_embeddings(saved_graph_info_dir)
    cluster_labels = perform_clustering(embeddings, num_clusters, use_gpu)
    save_cluster_assignments(output_dir, cluster_labels, graph_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings for graphs.")
    
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=analysis_constants.EMBEDDINGS_BASE_PATH,
        help="Path to the directory containing saved embeddings for graphs."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=analysis_constants.CLUSTER_OUTPUT_BASE_PATH,
        help="Path to save the clustering results."
    )
    parser.add_argument(
        "-k",
        "--num_clusters",
        type=int,
        default=8,
        help="Number of clusters for KMeans."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for clustering if available."
    )
    
    args = parser.parse_args()
    
    cluster_embeddings(
        saved_graph_info_dir=args.directory,
        output_dir=args.output,
        num_clusters=args.num_clusters,
        use_gpu=args.use_gpu
    )
