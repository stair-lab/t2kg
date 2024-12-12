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
API_KEY = "AIzaSyAGqzM57z8iyx4lp6-lsqY5tLMw6dPMzcE"
API_QUOTA_LIMIT = 1
API_QUOTA_MARGIN = 0.3
API_CUSTOM_LIMIT = API_QUOTA_LIMIT + API_QUOTA_MARGIN
PROMPT_MESSAGE_PREPENDED_TO_CLUSTERS = "I think: "
TRIPLE_NUMBER = "triple_number"
TRIPLE = "triple"
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


def save_results(results_to_be_saved, saved_new_triples_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_path = graph_utils.get_new_triples_analysis_save_path(saved_new_triples_path, output_dir)
    with open(results_path, 'w') as f:
        json.dump(results_to_be_saved, f, indent=4)

# Request function
def analyze_triple(triple):
    text_to_analyze = f"I think: {' '.join(triple)}."
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


def get_triples_toxicity_metrics(new_triples):
    results = []
    for i, triple in enumerate(new_triples):
        print(f"Analyzing Triple {i+1}/{len(new_triples)}. Triple: {triple}")
        result = analyze_triple(triple)
        if result:
            scores = {
                TRIPLE_NUMBER: i,
                TRIPLE: triple,
                TOXICITY: result[ATTRIBUTE_SCORES][TOXICITY][SUMMARY_SCORE][VALUE],
                SEVERE_TOXICITY: result[ATTRIBUTE_SCORES][SEVERE_TOXICITY][SUMMARY_SCORE][VALUE],
                INSULT: result[ATTRIBUTE_SCORES][INSULT][SUMMARY_SCORE][VALUE],
                IDENTITY_ATTACK: result[ATTRIBUTE_SCORES][IDENTITY_ATTACK][SUMMARY_SCORE][VALUE],
                PROFANITY: result[ATTRIBUTE_SCORES][PROFANITY][SUMMARY_SCORE][VALUE],
                THREAT: result[ATTRIBUTE_SCORES][THREAT][SUMMARY_SCORE][VALUE]
            }
            results.append(scores)
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


def new_triples_metrics_for_graph(saved_new_triples_path, output_dir):
    print(f"Running new triples analysis on {saved_new_triples_path}...")
    
    
    with open(saved_new_triples_path, 'r') as f:
        imputation_data = json.load(f)

    new_triples = imputation_data[analysis_constants.PREDICTED_TRIPLES_KEY]
    
    # Use a min-heap to track the top k triples
    triples_toxicity = get_triples_toxicity_metrics(new_triples)

    aggregate_toxicity_statistics = get_aggregate_toxicity_metrics(triples_toxicity)
    # here we want to compute aggregate statistics for the graph's toxicity
    
    results_to_be_saved = {
        analysis_constants.AGGREGATE_TOXICITY_STATS_KEY: aggregate_toxicity_statistics,
        analysis_constants.NEW_TRIPLES_TOXICITY_STATS_KEY: triples_toxicity
    }

    save_results(results_to_be_saved, saved_new_triples_path, output_dir)

def get_new_triples_metrics_for_graphs(selected, directory, output):
    if selected is None:
        selected = [d for d in os.listdir(directory)]

    for graph_name in selected:
        saved_new_triples_path = os.path.join(directory, graph_name)
        new_triples_metrics_for_graph(saved_new_triples_path, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embeddings for knowledge graphs.")
   
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=analysis_constants.IMPUTATION_BASE_PATH,  # Default placeholder value
        help="Path to the directory containing saved data for graphs. Default is './data'."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=analysis_constants.NEW_TRIPLES_ANALYSIS_BASE_PATH,  # Default placeholder value
        help="Path to save the visualizations."
    )

    args = parser.parse_args()
    get_new_triples_metrics_for_graphs(selected=None, directory=args.directory, output=args.output)