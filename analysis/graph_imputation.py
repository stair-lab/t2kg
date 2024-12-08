import argparse
import os
import analysis_constants
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
import itertools
import numpy as np
import torch
import json

K = 5

def save_results(results_to_be_saved, saved_graph_info_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_path = graph_utils.get_imputation_results_path(saved_graph_info_dir, output_dir)
    with open(results_path, 'w') as f:
        json.dump(results_to_be_saved, f, indent=4)

def run_imputation_on_graph(saved_graph_info_dir, output_dir, k):
    entity_embeddings, relation_embeddings, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)

    entities = graph_data[analysis_constants.ENTITIES_KEY]
    relations = graph_data[analysis_constants.RELATIONS_KEY]
    candidate_triples = list(itertools.product(entities, relations, entities))

    def score_triple(head, relation, tail):
        # Get embeddings
        head_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY][head]
        tail_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY][tail]
        relation_idx = graph_data[analysis_constants.RELATION_MAPPING_KEY][relation]

        head_emb = entity_embeddings[head_idx]
        tail_emb = entity_embeddings[tail_idx]
        relation_emb = relation_embeddings[relation_idx]

        distance = np.linalg.norm(head_emb + relation_emb - tail_emb)
        return -distance  # Negative distance so higher scores are better

    triple_scores = []
    for head, relation, tail in candidate_triples:
        # Skip existing triples
        if any(
            (existing_head == head and
            existing_relation == relation and
            existing_tail == tail) or (head == tail)
            for existing_head, existing_relation, existing_tail in graph_data[analysis_constants.TRIPLES_KEY]
        ):
            continue

        score = score_triple(head, relation, tail)
        triple_scores.append((head, relation, tail, score))
    top_triples = sorted(triple_scores, key=lambda x: x[3], reverse=True)[:k]

    print(f"Top {k} Predicted New Triples:")
    for head, relation, tail, score in top_triples:
        print(f"{head} --[{relation}]--> {tail} (Score: {score:.4f})")
    triples_to_be_saved = [(head, relation, tail) for head, relation, tail, _ in top_triples]
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