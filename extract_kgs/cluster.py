"""
This script implements a clustering algorithm for entities in a knowledge graph.
It uses GPT-4o to dynamically group entities cluster by cluster based on semantic similarity,
handling variations in tenses, plural forms, stem forms, and capitalization.
The process involves iteratively extracting clusters, validating them,
selecting representative entities, and continuing until all entities are processed.
This approach aims to reduce redundancy and improve the coherence of the knowledge graph
by grouping semantically equivalent or closely related entities.
"""

from typing import List, Dict, Set, Optional
import json
from pydantic import BaseModel, model_validator, Field
from openai import OpenAI
import logging
from index import client, MODEL
import random
import time
from openai import RateLimitError

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

MAX_RETRIES = 3
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
        if self.representative not in self.items:
            raise ValueError("Representative must be one of the items in the cluster")
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
    
    logger.debug(f"Attempting to extract {item_type} cluster from {len(items)} items")
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
            temperature=0
        )
        
        cluster = completion.choices[0].message.parsed.cluster
        if not cluster:
            logger.debug(f"No suitable {item_type} clusters found in batch")
        else:
            logger.info(f"Found {item_type} cluster with {len(cluster)} items: {cluster}")
            
        del ClusterResponse._valid_entities
        return cluster
        
    except Exception as e:
        logger.error(f"Error extracting cluster: {str(e)}")
        raise

@handle_rate_limit
def validate_cluster(cluster: List[str], item_type: str = "entities") -> List[str]:
    """Validate that cluster members are truly synonyms"""
    logger.debug(f"Validating {item_type} cluster with {len(cluster)} items")
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
        logger.debug(f"Validation result: {len(validated)}/{len(cluster)} items validated")
        return validated
    except Exception as e:
        logger.error(f"Error in cluster validation: {str(e)}")
        raise

@handle_rate_limit
def choose_representative(cluster: List[str]) -> str:
    """Choose the best entity name to represent the cluster"""
    logger.debug(f"Choosing representative for cluster with {len(cluster)} items")
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
            logger.debug(f"Chose representative: {representative}")
            return representative
        except ValueError as ve:
            logger.warning(f"Representative selection attempt {attempt + 1}/{max_retries} failed: {ve}")
            if attempt == max_retries - 1:
                logger.error("All attempts failed, falling back to heuristic")
                return min(cluster, key=lambda x: (len(x), x.lower()))
        except Exception as e:
            logger.error(f"Error choosing representative: {str(e)}")
            raise

class ClusterIndexResponse(BaseModel):
    cluster_indices: List[Optional[int]] = Field(description="A list of indices corresponding to the clusters each item can be added to, or None if they don't fit any existing cluster")

@handle_rate_limit
def check_items_for_existing_clusters(items: List[str], clusters: List[Cluster], item_type: str) -> List[Optional[int]]:
    """Check if multiple items can be added to any existing clusters"""
    logger.debug(f"Checking {len(items)} items against {len(clusters)} existing clusters")
    try:
        items_str = ", ".join(items)
        clusters_str = ", ".join([f"Cluster {i}: {c.items}" for i, c in enumerate(clusters)])
        
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"Determine if the given {item_type} can be added to any of the existing clusters. Return a list of indices corresponding to the clusters each item can be added to, or null if it doesn't fit any cluster."},
                {"role": "user", "content": f"{item_type.capitalize()}: {items_str}\nExisting clusters: {clusters_str}"},
            ],
            response_format=ClusterIndexResponse,
            temperature=0
        )
        
        cluster_indices = completion.choices[0].message.parsed.cluster_indices
        logger.debug(f"Found potential matches: {cluster_indices}")
        
        # Validate the new clusters
        for i, index in enumerate(cluster_indices):
            if index is not None:
                new_cluster = clusters[index].items + [items[i]]
                validated_cluster = validate_cluster(new_cluster, item_type)
                
                if len(validated_cluster) != len(clusters[index].items) + 1:
                    logger.debug(f"Validation failed for item {items[i]} in cluster {index}")
                    cluster_indices[i] = None
        
        return cluster_indices
      
    except Exception as e:
        logger.error(f"Error checking items for existing clusters: {str(e)}")
        raise

def cluster_items(items: Set[str], item_type: str = "entities") -> List[Cluster]:
    """Generic clustering function that works for both entities and edges"""
    remaining_items = list(items)
    clusters: List[Cluster] = []
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
        else:
            no_progress_count += 1
            logger.info(f"No valid cluster found. Attempts without progress: {no_progress_count}")
            # Randomly shuffle the remaining items
            random.shuffle(remaining_items)
            logger.info(f"Shuffled the remaining {len(remaining_items)} items")
            
            if no_progress_count >= 2:
                logger.warning(f"No progress for 2 iterations. Checking remaining {item_type} for existing clusters.")
                items_to_remove = []
                for i in range(0, len(remaining_items), 10):
                    batch = remaining_items[i:i+10]
                    cluster_indices = check_items_for_existing_clusters(batch, clusters, item_type)
                    for j, index in enumerate(cluster_indices):
                        if index is not None:
                            clusters[index].items.append(batch[j])
                            items_to_remove.append(batch[j])
                            logger.info(f"Added {batch[j]} to existing cluster: {clusters[index].items}")
                        else:
                            clusters.append(Cluster(
                                items=[batch[j]],
                                representative=batch[j]
                            ))
                            items_to_remove.append(batch[j])
                            logger.info(f"Created single-item cluster for {batch[j]}")
                remaining_items = [item for item in remaining_items if item not in items_to_remove]
                break
    return clusters

def process_file(input_file: str):
    """Process a single input file, clustering both entities and edges"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or not all(k in data for k in ['entities', 'edges']):
                raise KeyError("Input file must contain 'entities' and 'edges' fields")
            
            # entity_clusters = sorted(
            #   cluster_items(set(sorted(data['entities'])), "entities"), 
            #   key=lambda c: len(c.items), reverse=True
            # )
            edge_clusters = sorted(
              cluster_items(set(sorted(data['edges'])), "edges"), 
              key=lambda c: len(c.items), reverse=True
            )
            output = {
                'entity_clusters': [cluster.dict() for cluster in entity_clusters],
                'edge_clusters': [cluster.dict() for cluster in edge_clusters]
            }
            
            output_file = input_file.replace('.json', '_clusters.json')
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Created {len(entity_clusters)} entity clusters and {len(edge_clusters)} edge clusters")
            logger.info(f"Saved clusters to {output_file}")
            
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
        'kgs/harmless_base_rejected_test_50.json',
    ]
    
    for input_file in input_files:
        logger.info(f"Processing {input_file}")
        process_file(input_file)
