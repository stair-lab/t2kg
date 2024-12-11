"""
This script implements a clustering algorithm for entities in a knowledge graph.
It uses GPT-4o to dynamically group entities cluster by cluster based on semantic similarity,
handling variations in tenses, plural forms, stem forms, and capitalization.
The process involves iteratively extracting clusters, validating them,
selecting representative entities, and continuing until all entities are processed.
This approach aims to reduce redundancy and improve the coherence of the knowledge graph
by grouping semantically equivalent or closely related entities.
"""

from typing import List, Dict, Set, Optional, Tuple
import json
from pydantic import BaseModel, model_validator, Field
from openai import OpenAI
import logging
from index import client, MODEL
import random
import time
from openai import RateLimitError
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTRACT_ENTITIES_CLUSTER_PROMPT = """Find ONE cluster of related entities from this list.
A cluster should contain entities that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Or entities with close semantic meanings.

Return only if you find entities that clearly belong together.
If you can't find a clear cluster, return an empty list."""

VALIDATE_ENTITIES_CLUSTER_PROMPT = """Verify if these entities belong in the same cluster.
A cluster should contain entities that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Or entities with close semantic meanings.

Return the entities that you are confident belong together as a single cluster.
If you're not confident, return an empty list."""

EXTRACT_EDGES_CLUSTER_PROMPT = """Find ONE cluster of closely related predicates from this list.
A cluster should contain predicates that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Predicates are the relations between subject and object entities. Ensure that the predicates in the same cluster have very close semantic meanings to describe the relation between the same subject and object entities.

Return only if you find predicates that clearly belong together.
If you can't find a clear cluster, return an empty list."""

VALIDATE_EDGES_CLUSTER_PROMPT = """Verify if these predicates belong in the same cluster.
A cluster should contain predicates that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Predicates are the relations between subject and object entities. Ensure that the predicates in the same cluster have very close semantic meanings to describe the relation between the same subject and object entities.

Return the predicates that you are confident belong together as a single cluster.
If you're not confident, return an empty list."""

MAX_RETRIES = 10
BASE_DELAY = 1  # Start with 1 second
MAX_DELAY = 60  # Maximum delay of 60 seconds
JITTER_RANGE = 0.1  # 10% jitter

def handle_rate_limit(func):
    """Decorator to handle rate limit errors with exponential backoff and jitter"""
    def wrapper(*args, **kwargs):
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                retry_count += 1
                if retry_count == MAX_RETRIES:
                    logger.error(f"Max retries ({MAX_RETRIES}) exceeded for rate limit")
                    raise Exception("Rate limit retries exceeded")
                
                # Calculate base delay with exponential backoff
                delay = min(BASE_DELAY * (2 ** retry_count), MAX_DELAY)
                
                # Add randomized jitter
                jitter = random.uniform(-JITTER_RANGE * delay, JITTER_RANGE * delay)
                final_delay = delay + jitter
                
                logger.warning(
                    f"Rate limit hit. Attempt {retry_count}/{MAX_RETRIES}. "
                    f"Waiting {final_delay:.2f} seconds before retry"
                )
                time.sleep(final_delay)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
    return wrapper

class Cluster(BaseModel):
    """Generic cluster class that works for both entities and edges"""
    items: List[str]
    representative: str
    
    @model_validator(mode='after')
    def validate_representative(self) -> 'Cluster':
        if not self.items:
            raise ValueError("Cluster must contain at least one item")
        # if self.representative not in self.items:
        #     raise ValueError("Representative must be one of the items in the cluster")
        return self

class ClusterResponse(BaseModel):
    cluster: List[str]
    @model_validator(mode='after')
    def validate_entities_exist(self) -> 'ClusterResponse':
        # This will be populated with the valid entities list before parsing
        if not hasattr(self, '_valid_entities'):
            return self
        
        self.cluster = [e for e in self.cluster if e in self._valid_entities]
        return self

class RepresentativeResponse(BaseModel):
    representative: str

@handle_rate_limit
def extract_single_cluster(items: List[str], item_type: str = "entities") -> List[str]:
    """Extract a single cluster from the list of items"""
    items_text = "\n".join([f"- {item}" for item in items])
    items_set = set(items)
    
    logger.info(f"Attempting to extract {item_type} cluster from {len(items)} items")
    try:
        ClusterResponse._valid_entities = items_set
        
        prompt = EXTRACT_ENTITIES_CLUSTER_PROMPT if item_type == "entities" else EXTRACT_EDGES_CLUSTER_PROMPT

        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": items_text},
            ],
            response_format=ClusterResponse,
            temperature=0,
	    max_tokens=1000
        )
       	print(f'completion choices: {completion}') 
        cluster = completion.choices[0].message.parsed.cluster
        if not cluster:
            logger.info(f"No suitable {item_type} clusters found in batch")
        else:
            logger.info(f"Found {item_type} cluster with {len(cluster)} items: {cluster}")
            
        del ClusterResponse._valid_entities
        return cluster
        
    except Exception as e:
        logger.error(f"Error extracting cluster: {str(e)}")
        return []

@handle_rate_limit
def validate_cluster(cluster: List[str], item_type: str = "entities") -> List[str]:
    """Validate that cluster members are truly synonyms"""
    logger.info(f"Validating {item_type} cluster with {len(cluster)} items")
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": VALIDATE_ENTITIES_CLUSTER_PROMPT if item_type == "entities" else VALIDATE_EDGES_CLUSTER_PROMPT},
                {"role": "user", "content": f"{item_type.capitalize()}: {', '.join(cluster)}"},
            ],
            response_format=ClusterResponse,
            temperature=0
        )
        validated = completion.choices[0].message.parsed.cluster
        logger.info(f"Validation result: {len(validated)}/{len(cluster)} items validated")
        return validated
    except Exception as e:
        logger.error(f"Error in cluster validation: {str(e)}")
        return []

@handle_rate_limit
def choose_representative(cluster: List[str]) -> str:
    """Choose the best entity name to represent the cluster"""
    logger.info(f"Choosing representative for cluster with {len(cluster)} items")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": """Select the best entity name in this cluster to represent the cluster.
Consider the following criteria:
- Prefer shorter names
- Prefer lowercase versions
- Choose the most general or commonly used form
- Avoid overly specific or rare variations"""},
                    {"role": "user", "content": f"Cluster: {', '.join(cluster)}"},
                ],
                response_format=RepresentativeResponse,
                temperature=0
            )
            representative = completion.choices[0].message.parsed.representative
            if representative not in cluster:
                raise ValueError(f"Chosen representative '{representative}' not in cluster")
            logger.info(f"Chose representative: {representative}")
            return representative
        except ValueError as ve:
            logger.warning(f"Representative selection attempt {attempt + 1}/{max_retries} warning: {ve}")
            if attempt == max_retries - 1:
                logger.error(f"All attempts failed, using representative that is not in cluster: representative={representative}")
                return representative
        except Exception as e:
            logger.error(f"Error choosing representative: {str(e)}")
            return min(cluster, key=lambda x: (len(x), x.lower()))

class ClusterIndexResponse(BaseModel):
    cluster_indices: List[Optional[int]] = Field(description="A list of indices corresponding to the clusters each item can be added to, or None if they don't fit any existing cluster")
    

@handle_rate_limit
def check_items_for_existing_clusters(items: List[str], clusters: List[Cluster], item_type: str) -> List[Optional[int]]:
    """Check if multiple items can be added to any existing clusters"""
    logger.info(f"Checking {len(items)} items against {len(clusters)} existing clusters")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            items_str = ", ".join(items)
            clusters_str = ", ".join([f"Cluster {i}: {c.items}" for i, c in enumerate(clusters)])
            
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": f"Determine if the given {item_type} can be added to any of the existing clusters. Return a list of indices corresponding to the clusters each item can be added to, or null if it doesn't fit any cluster."},
                    {"role": "user", "content": f"List of {len(items)} {item_type.capitalize()} to check: {items_str}\nExisting clusters: {clusters_str}"},
                ],
                response_format=ClusterIndexResponse,
                temperature=0
            )
            
            cluster_indices = completion.choices[0].message.parsed.cluster_indices
            logger.info(f"Found potential matches: {cluster_indices}")
            
            # Check for exceptions in the cluster_indices
            if len(cluster_indices) != len(items) or any(index is not None and (index < 0 or index >= len(clusters)) for index in cluster_indices):
                logger.warning(f"Found {len(cluster_indices)} potential matches for {len(items)} items")
                raise ValueError("Invalid cluster index found in response")
            
            # Validate the new clusters
            for i, index in enumerate(cluster_indices):
                if index is not None:
                    new_cluster = clusters[index].items + [items[i]]
                    validated_cluster = validate_cluster(new_cluster, item_type)
                    
                    if len(validated_cluster) != len(clusters[index].items) + 1:
                        logger.info(f"Validation failed for item {items[i]} in cluster {index}")
                        cluster_indices[i] = None
                elif index is not None:
                    logger.info(f"Index {index} is out of range for clusters list or {i} is out of range for items list. Skipping.")
                    cluster_indices[i] = None
            
            return cluster_indices
        except Exception as e:
            logger.error(f"Error checking items for existing clusters (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error("All attempts failed. Returning empty list.")
                return [None] * len(items)

def save_progress(output_file: str, clusters: List[Cluster], remaining_items: List[str], item_type: str, status: str = "in_progress"):
    """Save the current clustering progress to a file"""
    try:
        # Read existing data if file exists
        data = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
        
        # Update with new data
        cluster_key = f"{item_type}_clusters"
        remaining_key = f"remaining_{item_type}"
        data[cluster_key] = [cluster.dict() for cluster in clusters]
        data[remaining_key] = remaining_items
        data["status"] = status
        
        # Write back to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved progress to {output_file}: {len(clusters)} clusters, {len(remaining_items)} remaining items")
    except Exception as e:
        logger.error(f"Error saving progress to {output_file}: {e}")

def load_progress(output_file: str, item_type: str) -> Tuple[List[Cluster], List[str], str]:
    """Load clustering progress from a file"""
    try:
        if not os.path.exists(output_file):
            return [], [], "in_progress"
            
        with open(output_file, 'r') as f:
            data = json.load(f)
            
        cluster_key = f"{item_type}_clusters"
        remaining_key = f"remaining_{item_type}"
        
        if cluster_key not in data or remaining_key not in data:
            return [], [], "in_progress"
            
        clusters = [Cluster(**cluster_data) for cluster_data in data[cluster_key]]
        remaining_items = data[remaining_key]
        status = data.get("status", "in_progress")
        
        logger.info(f"Loaded progress from {output_file}: {len(clusters)} clusters, {len(remaining_items)} remaining items")
        return clusters, remaining_items, status
    except Exception as e:
        logger.error(f"Error loading progress from {output_file}: {e}")
        return [], [], "in_progress"

def cluster_items(items: Set[str], output_file: str, item_type: str = "entities") -> List[Cluster]:
    """Generic clustering function that works for both entities and edges"""
    # Load existing progress if any
    clusters, loaded_remaining_items, status = load_progress(output_file, item_type)
    
    # If the clustering is already done, return the existing clusters
    if status == "done":
        logger.info(f"Clustering already completed for {output_file}. Skipping.")
        return clusters
    
    # If we have existing progress and it's not done, use it
    if status == "in_progress" and loaded_remaining_items:
        remaining_items = loaded_remaining_items
    else:
        # If no progress or completed, start fresh
        remaining_items = list(items)
        clusters = []
        save_progress(output_file, clusters, remaining_items, item_type)
    
    iteration = 0
    no_progress_count = 0
    
    while remaining_items:
        iteration += 1
        logger.info(f"Iteration {iteration}: Remaining {item_type} to cluster: {len(remaining_items)}")
        
        suggested_cluster = extract_single_cluster(remaining_items, item_type)
        validated_cluster = validate_cluster(suggested_cluster, item_type) if suggested_cluster else []
        
        if len(validated_cluster) >= 2:
            no_progress_count = 0
            representative = choose_representative(validated_cluster)
            clusters.append(Cluster(
                items=validated_cluster,
                representative=representative
            ))
            
            remaining_items = [item for item in remaining_items if item not in validated_cluster]
            logger.info(f"Created {item_type} cluster with {len(validated_cluster)} items: {validated_cluster}")
            logger.info(f"Representative: {representative}")
            
            # Save progress after each new cluster
            save_progress(output_file, clusters, remaining_items, item_type)
        else:
            no_progress_count += 1
            logger.info(f"No valid cluster found. Attempts without progress: {no_progress_count}")
            # Randomly shuffle the remaining items
            random.shuffle(remaining_items)
            logger.info(f"Shuffled the remaining {len(remaining_items)} items")
            
            if no_progress_count >= 10:
                logger.warning(f"No progress for 10 iterations. Checking remaining {item_type} for existing clusters.")
                items_to_remove = []
                
                # Process remaining items in batches of 10
                for i in range(0, len(remaining_items), 10):
                    batch = remaining_items[i:i+10]
                    cluster_indices = check_items_for_existing_clusters(batch, clusters, item_type)
                    
                    if not cluster_indices:
                        logger.warning(f"Failed to get cluster indices for batch {i//10 + 1}")
                        continue
                        
                    # Process each item in the batch
                    for j, index in enumerate(cluster_indices):
                        try: # This try catch for debugging
                            item = batch[j]
                            
                            if index is not None:
                                # Add to existing cluster after validation
                                new_cluster = clusters[index].items + [item]
                                validated_cluster = validate_cluster(new_cluster, item_type)
                                
                                if len(validated_cluster) == len(new_cluster):
                                    clusters[index].items = validated_cluster
                                    items_to_remove.append(item)
                                    logger.info(f"Added {item} to existing cluster {index}")
                            else:
                                # Create new single-item cluster
                                new_cluster = Cluster(
                                    items=[item],
                                    representative=item
                                )
                                clusters.append(new_cluster)
                                items_to_remove.append(item)
                                logger.info(f"Created new single-item cluster for {item}")
                                
                        except Exception as e:
                            logger.error(f"Error processing item {j} in batch {i//10 + 1}: {str(e)}")
                    
                    # Save progress after each batch
                    remaining_items = [item for item in remaining_items if item not in items_to_remove]
                    save_progress(output_file, clusters, remaining_items, item_type)
                
                logger.info(f"Processed all remaining items. {len(items_to_remove)} items processed.")
                break
    
    # Mark as complete
    save_progress(output_file, clusters, [], item_type, status="done")
    return clusters

def process_file(input_file: str):
    """Process a single input file, clustering both entities and edges"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or not all(k in data for k in ['entities', 'edges']):
                raise KeyError("Input file must contain 'entities' and 'edges' fields")
            
            # Generate output filenames
            base_output = input_file.replace('.json', '_clusters.json')
            entity_output = input_file.replace('.json', '_entity_clusters.json')
            edge_output = input_file.replace('.json', '_edge_clusters.json')
            
            entity_clusters = sorted(
                cluster_items(set(sorted(data['entities'])), entity_output, "entities"),
                key=lambda c: len(c.items), reverse=True
            )
            edge_clusters = sorted(
                cluster_items(set(sorted(data['edges'])), edge_output, "edges"),
                key=lambda c: len(c.items), reverse=True
            )
            
            # Save final combined output
            output = {
                'entity_clusters': [cluster.dict() for cluster in entity_clusters],
                'edge_clusters': [cluster.dict() for cluster in edge_clusters],
                'status': 'done'
            }
            
            with open(base_output, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Created {len(entity_clusters)} entity clusters and {len(edge_clusters)} edge clusters")
            logger.info(f"Saved final clusters to {base_output}")
            
            # Print statistics for both types
            for cluster_type, clusters in [("Entity", entity_clusters), ("Edge", edge_clusters)]:
                sizes = [len(cluster.items) for cluster in clusters]
                if sizes:
                    logger.info(f"\n{cluster_type} cluster statistics:")
                    logger.info(f"  Min: {min(sizes)}")
                    logger.info(f"  Max: {max(sizes)}")
                    logger.info(f"  Average: {sum(sizes)/len(sizes):.2f}")
                    logger.info(f"  Singles: {sum(1 for size in sizes if size == 1)}")
                    logger.info(f"  Multiples: {sum(1 for size in sizes if size > 1)}")
                
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")

if __name__ == "__main__":
    input_files = [
        'kgs/224w_final_deliverable/cleaned_300_pku-safe-30k-test-Mistral-7B-Instruct-v0.2.json',
        'kgs/224w_final_deliverable/advil.json',
        # 'kgs/224w_final_deliverable/cleaned_300_pku-safe-30k-test-gemma-2-9b-it.json',
        # 'kgs/224w_final_deliverable/cleaned_300_pku-safe-30k-test-Mistral-7B-v0.2_no_ann.json',
        # 'kgs/224w_final_deliverable/cleaned_300_pku-safe-30k-gemma-2-9b_no_ann.json',
        # 'kgs/224w_final_deliverable/cleaned_300_pku-safe-30k-test-Mistral-7B-Instruct-v0.2.json',
    ]
    
    for input_file in input_files:
        logger.info(f"Processing {input_file}")
        process_file(input_file)
