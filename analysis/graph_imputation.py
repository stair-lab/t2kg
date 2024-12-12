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

PROPORTION_OF_RELATIONS_TO_SAMPLE = 0.1


def save_results(results_to_be_saved, saved_graph_info_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_path = graph_utils.get_imputation_results_path(saved_graph_info_dir, output_dir)
    with open(results_path, 'w') as f:
        json.dump(results_to_be_saved, f, indent=4)

def run_imputation_on_graph(saved_graph_info_dir, output_dir, k):
    print(f"Running imputation on {saved_graph_info_dir}")
    entity_embeddings, relation_embeddings, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)

    # Move embeddings to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entity_embeddings = torch.tensor(entity_embeddings, device=device, dtype=torch.float32)
    relation_embeddings = torch.tensor(relation_embeddings, device=device, dtype=torch.float32)

    entities = graph_data[analysis_constants.ENTITIES_KEY]
    relations = graph_data[analysis_constants.RELATIONS_KEY]
    existing_triples = set(
        (head, relation, tail) for head, relation, tail in graph_data[analysis_constants.TRIPLES_KEY]
    )

    clusters = graph_data[analysis_constants.ENTITY_CLUSTERS_KEY]

    cluster_to_entities = {}
    for entity, cluster_label in zip(entities, clusters):
        cluster_to_entities.setdefault(cluster_label, []).append(entity)

    def score_triple(head, relation, tail):
        # Get embeddings
        head_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY][head]
        tail_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY][tail]
        relation_idx = graph_data[analysis_constants.RELATION_MAPPING_KEY][relation]

        head_emb = entity_embeddings[head_idx]
        tail_emb = entity_embeddings[tail_idx]
        relation_emb = relation_embeddings[relation_idx]

        distance = torch.norm(head_emb + relation_emb - tail_emb)
        return -distance.item()  # Negative distance so higher scores are better

    # Use a min-heap to track the top k triples
    heap = []
    num_relations_to_sample = int(len(relations) * PROPORTION_OF_RELATIONS_TO_SAMPLE)

    i = 0
    for cluster_label, cluster_entities in cluster_to_entities.items():
        print(f"Processing {i+1}/{len(cluster_to_entities)} cluster, cluster label: {cluster_label} with {len(cluster_entities)} entities")
        for head, tail in itertools.product(cluster_entities, repeat=2):
            if head == tail:  # Skip invalid triples where head == tail
                continue
            # Randomly select K relations
            selected_relations = random.sample(relations, min(len(relations), num_relations_to_sample))
            for relation in selected_relations:
                if (head, relation, tail) in existing_triples:
                    continue
                score = score_triple(head, relation, tail)
                if len(heap) < k:
                    heapq.heappush(heap, (score, head, relation, tail))
                else:
                    heapq.heappushpop(heap, (score, head, relation, tail))
        i += 1

    # Extract the top k triples
    top_triples = sorted(heap, key=lambda x: x[0], reverse=True)

    print(f"Top {k} Predicted New Triples:")
    for score, head, relation, tail in top_triples:
        print(f"{head} --[{relation}]--> {tail} (Score: {-score:.4f})")

    triples_to_be_saved = [(head, relation, tail) for _, head, relation, tail in top_triples]
    results_to_be_saved = {
        analysis_constants.ORIGINAL_TRIPLES_KEY: graph_data[analysis_constants.TRIPLES_KEY],
        analysis_constants.PREDICTED_TRIPLES_KEY: triples_to_be_saved
    }

    save_results(results_to_be_saved, saved_graph_info_dir, output_dir)

def impute_graphs(selected, directory, output):
    if selected is None:
        selected = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for graph_name in selected:
        saved_graph_info_dir = os.path.join(directory, graph_name)
        run_imputation_on_graph(saved_graph_info_dir, output, analysis_constants.K_NEW_TRIPLES)

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
        default=analysis_constants.IMPUTATION_BASE_PATH,  # Default placeholder value
        help="Path to save the visualizations."
    )

    args = parser.parse_args()
    impute_graphs(selected=None, directory=args.directory, output=args.output)