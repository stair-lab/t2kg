import argparse
import os
import analysis_constants
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
import itertools
import numpy as np
import json
import torch
import heapq
import torch
import random
from googleapiclient import discovery
import requests
import time
import numpy as np
from scipy.stats import skew, kurtosis

# Your Perspective API key
API_KEY = ""
API_QUOTA_LIMIT = 1
API_QUOTA_MARGIN = 0.3
API_CUSTOM_LIMIT = API_QUOTA_LIMIT + API_QUOTA_MARGIN
PROMPT_MESSAGE_PREPENDED_TO_CLUSTERS = "I think the following entities have a close relationship and they are related: "
CLUSTER_NUMBER = "cluster_number"
CLUSTER_ENTITIES = "cluster_entities"
TOXICITY = "TOXICITY"
SEVERE_TOXICITY = "SEVERE_TOXICITY"
INSULT = "INSULT"
IDENTITY_ATTACK = "IDENTITY_ATTACK"
PROFANITY = "PROFANITY"
THREAT = "THREAT"
SUMMARY_SCORE = "summaryScore"
VALUE = "value"
ATTRIBUTE_SCORES = "attributeScores"

def save_results(results_to_be_saved, saved_graph_info_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_path = graph_utils.get_cluster_states_results_save_path(saved_graph_info_dir, output_dir)
    with open(results_path, 'w') as f:
        json.dump(results_to_be_saved, f, indent=4)


def get_graph_toxicity_metric(saved_graph_info_dir, output_dir, k):
    print(f"Running toxicity analysis on {saved_graph_info_dir}...")
    entity_embeddings, relation_embeddings, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)

    # Move embeddings to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entity_embeddings = torch.tensor(entity_embeddings, device=device, dtype=torch.float32)
    relation_embeddings = torch.tensor(relation_embeddings, device=device, dtype=torch.float32)

    entities = graph_data[analysis_constants.ENTITIES_KEY]

    clusters = graph_data[analysis_constants.ENTITY_CLUSTERS_KEY]
    cluster_to_entities = {}
    for entity, cluster_label in zip(entities, clusters):
        cluster_to_entities.setdefault(cluster_label, []).append(entity)

    # print("The clusters are: ", cluster_to_entities)

    mean_size = sum(len(cluster) for cluster in cluster_to_entities.values()) / len(cluster_to_entities)
    max_size = max(len(cluster) for cluster in cluster_to_entities.values())
    min_size = min(len(cluster) for cluster in cluster_to_entities.values())
    std_size = np.std([len(cluster) for cluster in cluster_to_entities.values()])

    # here we want to compute aggregate statistics for the graph's toxicity
    
    results_to_be_saved = {
        analysis_constants.CLUSTER_STATS_KEY_MEAN_SIZE: mean_size,
        analysis_constants.CLUSTER_STATS_KEY_MAX_SIZE: max_size,
        analysis_constants.CLUSTER_STATS_KEY_MIN_SIZE: min_size,
        analysis_constants.CLUSTER_STATS_KEY_STD_SIZE: std_size
    }

    save_results(results_to_be_saved, saved_graph_info_dir, output_dir)

def get_toxicity_metrics(selected, directory, output):
    if selected is None:
        selected = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for graph_name in selected:
        saved_graph_info_dir = os.path.join(directory, graph_name)
        get_graph_toxicity_metric(saved_graph_info_dir, output, analysis_constants.K_NEW_TRIPLES)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embeddings for knowledge graphs.")
   
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=analysis_constants.EMBEDDINGS_BASE_PATH,  # Default placeholder value
        help="Path to the directory containing saved data for graphs. Default is './data'."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=analysis_constants.CLUSTER_METRICS_BASE_PATH,  # Default placeholder value
        help="Path to save the visualizations."
    )

    args = parser.parse_args()
    get_toxicity_metrics(selected=None, directory=args.directory, output=args.output)