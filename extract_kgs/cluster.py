"""
This script implements a clustering algorithm for entities in a knowledge graph.
It uses GPT-4o to dynamically group entities cluster by cluster based on semantic similarity,
handling variations in tenses, plural forms, stem forms, and capitalization.
The process involves iteratively extracting clusters, validating them,
selecting representative entities, and continuing until all entities are processed.
This approach aims to reduce redundancy and improve the coherence of the knowledge graph
by grouping semantically equivalent or closely related entities.
"""

from typing import List, Dict, Set
import json
from pydantic import BaseModel, model_validator
from openai import OpenAI
import logging
from index import client, MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTRACT_ENTITIES_CLUSTER_PROMPT = """Find ONE cluster of closely related entities from this list.
A cluster should contain entities that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Or entities with very close semantic meanings.

Be conservative - when in doubt, keep entities separate.
Return only if you find entities that clearly belong together.
If you can't find a clear cluster, return an empty list."""

VALIDATE_ENTITIES_CLUSTER_PROMPT = """Verify if these entities belong in the same cluster.
A cluster should contain entities that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Or entities with very close semantic meanings.

Be conservative.
Return the entities that you are completely confident belong together as a single cluster.
If you're not confident about the equivalence, return an empty list."""

EXTRACT_EDGES_CLUSTER_PROMPT = """Find ONE cluster of closely related predicates from this list.
A cluster should contain predicates that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Predicates are the relations between subject and object entities. Ensure that the predicates in the same cluster have very close semantic meanings to describe the relation between the same subject and object entities.

Be conservative - when in doubt, keep predicates separate.
Return only if you find predicates that clearly belong together.
If you can't find a clear cluster, return an empty list."""

VALIDATE_EDGES_CLUSTER_PROMPT = """Verify if these predicates belong in the same cluster.
A cluster should contain predicates that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Predicates are the relations between subject and object entities. Ensure that the predicates in the same cluster have very close semantic meanings to describe the relation between the same subject and object entities.

Be conservative.
Return the predicates that you are completely confident belong together as a single cluster.
If you're not confident about the equivalence, return an empty list."""


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
            
        invalid_entities = [e for e in self.cluster if e not in self._valid_entities]
        if invalid_entities:
            raise ValueError(f"Invalid entities in cluster: {invalid_entities}")
        return self

class RepresentativeResponse(BaseModel):
    representative: str

def extract_single_cluster(items: List[str], item_type: str = "entities") -> List[str]:
    """Extract a single cluster from the list of items"""
    items_text = "\n".join([f"- {item}" for item in items])
    items_set = set(items)
    
    max_retries = 3
    invalid_items = []
    for attempt in range(max_retries):
        try:
            ClusterResponse._valid_entities = items_set  # Reuse existing validation
            
            prompt = EXTRACT_ENTITIES_CLUSTER_PROMPT if item_type == "entities" else EXTRACT_EDGES_CLUSTER_PROMPT

            if invalid_items:
                prompt += f"\n\nNote: The following were invalid in previous attempt(s): {', '.join(invalid_items)}."
            
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
                logger.info(f"Model found no suitable {item_type} clusters in this batch")
            else:
                logger.info(f"Model suggested {item_type} cluster: {cluster}")
                
            del ClusterResponse._valid_entities
            return cluster
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("All attempts failed to extract cluster")
                return []
    
    return []

def validate_cluster(cluster: List[str], item_type: str = "entities") -> List[str]:
    """Validate that cluster members are truly synonyms"""
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
        return completion.choices[0].message.parsed.cluster
    except Exception as e:
        logger.error(f"Error in cluster validation: {e}")
        return []

def choose_representative(cluster: List[str]) -> str:
    """Choose the best entity name to represent the cluster"""
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
            # Validate that the chosen representative is in the cluster
            if representative not in cluster:
                raise ValueError(f"Chosen representative '{representative}' not in cluster")
            return representative
        except ValueError as ve:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {ve}")
            if attempt == max_retries - 1:
                logger.error("All attempts to choose representative failed, falling back to heuristic")
                return min(cluster, key=lambda x: (len(x), x.lower()))
        except Exception as e:
            logger.error(f"Error in choosing representative: {e}")
            return min(cluster, key=lambda x: (len(x), x.lower()))

def cluster_items(items: Set[str], item_type: str = "entities") -> List[Cluster]:
    """Generic clustering function that works for both entities and edges"""
    remaining_items = items.copy()
    clusters: List[Cluster] = []
    iteration = 0
    no_progress_count = 0
    
    while remaining_items:
        iteration += 1
        logger.info(f"Iteration {iteration}: Remaining {item_type} to cluster: {len(remaining_items)}")
        
        suggested_cluster = extract_single_cluster(list(remaining_items), item_type)
        validated_cluster = suggested_cluster # TODO: Comment this
        # validated_cluster = validate_cluster(suggested_cluster, item_type) if suggested_cluster else []
        
        if len(validated_cluster) >= 2:
            no_progress_count = 0
            representative = choose_representative(validated_cluster)
            clusters.append(Cluster(
                items=validated_cluster,
                representative=representative
            ))
            
            remaining_items -= set(validated_cluster)
            logger.info(f"Created {item_type} cluster with {len(validated_cluster)} items: {validated_cluster}")
            logger.info(f"Representative: {representative}")
        else:
            no_progress_count += 1
            logger.info(f"No valid cluster found. Attempts without progress: {no_progress_count}")
            
            if no_progress_count >= 3:
                logger.warning(f"No progress for 3 iterations. Processing all remaining {item_type} as singles. Remaining {item_type}: {remaining_items}")
                for item in remaining_items:
                    clusters.append(Cluster(
                        items=[item],
                        representative=item
                    ))
                break
    
    return clusters

def process_file(input_file: str):
    """Process a single input file, clustering both entities and edges"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or not all(k in data for k in ['entities', 'edges']):
                raise KeyError("Input file must contain 'entities' and 'edges' fields")
            
            entity_clusters = sorted(cluster_items(set(data['entities']), "entities"), key=lambda c: len(c.items), reverse=True)
            edge_clusters = sorted(cluster_items(set(data['edges']), "edges"), key=lambda c: len(c.items), reverse=True)
            
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
