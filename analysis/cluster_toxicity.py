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


# Perspective API endpoint
PERSPECTIVE_API_URL = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)


def save_results(results_to_be_saved, saved_graph_info_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_path = graph_utils.get_toxicity_results_save_path(saved_graph_info_dir, output_dir)
    with open(results_path, 'w') as f:
        json.dump(results_to_be_saved, f, indent=4)

# Request function
def analyze_cluster(cluster):
    text_to_analyze = f"The following entities are closely related entities: {', '.join(cluster)}."
    analyze_request = {
        "comment": {"text": text_to_analyze},
        "languages": ["en"],
        "requestedAttributes": {
            TOXICITY: {},
            SEVERE_TOXICITY: {},
            INSULT: {}, 
            IDENTITY_ATTACK: {}, 
            PROFANITY: {}, 
            THREAT: {}
        }
    }
    # Send POST request
    response = client.comments().analyze(body=analyze_request).execute()
    # print(json.dumps(response, indent=2))
    return response


def get_graph_cluster_toxicity_metrics(cluster_to_entities):
    results = []
    i = 1
    for c, cluster in cluster_to_entities.items():
        print(f"Analyzing Cluster {i}/{len(cluster_to_entities)}. Cluster index: {c}")
        result = analyze_cluster(cluster)
        if result:
            scores = {
                CLUSTER_NUMBER: i,
                CLUSTER_ENTITIES: cluster,
                TOXICITY: result[ATTRIBUTE_SCORES][TOXICITY][SUMMARY_SCORE][VALUE],
                SEVERE_TOXICITY: result[ATTRIBUTE_SCORES][SEVERE_TOXICITY][SUMMARY_SCORE][VALUE],
                INSULT: result[ATTRIBUTE_SCORES][INSULT][SUMMARY_SCORE][VALUE],
                IDENTITY_ATTACK: result[ATTRIBUTE_SCORES][IDENTITY_ATTACK][SUMMARY_SCORE][VALUE],
                PROFANITY: result[ATTRIBUTE_SCORES][PROFANITY][SUMMARY_SCORE][VALUE],
                THREAT: result[ATTRIBUTE_SCORES][THREAT][SUMMARY_SCORE][VALUE]
            }
            results.append(scores)
        i += 1
        time.sleep(API_CUSTOM_LIMIT)

    return results

# Function to calculate the statistics
def get_aggregate_toxicity_metrics(results):
    toxicity_stats = {
        TOXICITY: {},
        SEVERE_TOXICITY: {},
        INSULT: {},
        IDENTITY_ATTACK: {},
        PROFANITY: {},
        THREAT: {}
    }

    # Initialize a dictionary for each toxicity type
    for metric in toxicity_stats:
        scores = [score[metric] for score in results]

        # Calculate the required metrics
        toxicity_stats[metric]['Mean'] = np.mean(scores)
        toxicity_stats[metric]['Median'] = np.median(scores)
        toxicity_stats[metric]['Standard Deviation'] = np.std(scores)
        toxicity_stats[metric]['Minimum'] = np.min(scores)
        toxicity_stats[metric]['Maximum'] = np.max(scores)
        toxicity_stats[metric]['Skewness'] = skew(scores)
        toxicity_stats[metric]['Kurtosis'] = kurtosis(scores)

    return toxicity_stats


def get_graph_toxicity_metric(saved_graph_info_dir, output_dir, k):
    print(f"Running toxicity analysis on {saved_graph_info_dir}...")
    entity_embeddings, relation_embeddings, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)

    # Move embeddings to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entity_embeddings = torch.tensor(entity_embeddings, device=device, dtype=torch.float32)
    relation_embeddings = torch.tensor(relation_embeddings, device=device, dtype=torch.float32)

    entities = graph_data[analysis_constants.ENTITIES_KEY]
    # _ = graph_data[analysis_constants.RELATIONS_KEY]
    # _ = set(
    #     (head, relation, tail) for head, relation, tail in graph_data[analysis_constants.TRIPLES_KEY]
    # )

    clusters = graph_data[analysis_constants.ENTITY_CLUSTERS_KEY]
    cluster_to_entities = {}
    for entity, cluster_label in zip(entities, clusters):
        cluster_to_entities.setdefault(cluster_label, []).append(entity)

    # print("The clusters are: ", cluster_to_entities)

    # Use a min-heap to track the top k triples
    clusters_toxicity = get_graph_cluster_toxicity_metrics(cluster_to_entities)

    aggregate_toxicity_statistics = get_aggregate_toxicity_metrics(clusters_toxicity)

    # here we want to compute aggregate statistics for the graph's toxicity
    
    results_to_be_saved = {
        analysis_constants.AGGREGATE_TOXICITY_STATS_KEY: aggregate_toxicity_statistics,
        analysis_constants.CLUSTER_TOXICITY_STATS_KEY: clusters_toxicity
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
        default=analysis_constants.CLUSTER_TOXICITY_BASE_PATH,  # Default placeholder value
        help="Path to save the visualizations."
    )

    args = parser.parse_args()
    get_toxicity_metrics(selected=None, directory=args.directory, output=args.output)