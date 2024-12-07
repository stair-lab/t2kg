# Extract Knowledge Graphs

This directory contains scripts for extracting knowledge graphs from text data.
- `index.py`: Extract entities and relations from text using GPT-4o.
- `aggregate.py`: Aggregate data from multiple JSON Lines (jsonl) files into a single dataset.
- `cluster.py`: Cluster entities using GPT-4o and create an entity enumeration.
- `visualize.py`: Visualize knowledge graphs using NetworkX and Matplotlib.

## Run

You'll need to set the `OPENAI_API_KEY` environment variable to your OpenAI API key before running the scripts.
Also, be sure to change the file path variables in each script to make sure they point to the correct input and output paths. (sorry for the inconvenience)

Run in this order:
- `python index.py`
- `python aggregate.py`
- `python cluster.py`
- `python visualize.py`

## About

## Extraction Process

The knowledge graph extraction process involves several steps, utilizing different scripts to handle entity and relation extraction, data aggregation, and entity and edge clustering. Here's a detailed breakdown of the process:

### 1. Entity and Relation Extraction (`index.py`):

This script uses OpenAI's Structured Output feature to extract entities and relations in a JSON schema from text data. We use GPT-4o-2024-08-06. 

The JSON schema is defined as follows:

```python
class KnowledgeGraphEntities(BaseModel):
    entities: List[str]
```

It processes input files containing user queries and assistant responses.
For each sample:
a. Extracts entities from the user query. 
b. Extracts additional entities from the assistant response.
c. Given a list of all sample entities and the assistant response string, extracts relations.

The extracted data is stored in a `GoldLabelDataset` structure, which includes:
- Original user query and assistant response
- Extracted entities from query
- Extracted entities from response
- Extracted relations (subject-predicate-object triples)

Note that we looked at spaCy and textacy for entity extraction, but ultimately decided to use GPT-4o for this task since it allowed for richer entities and relations to be extracted.

### 2. Data Aggregation (`aggregate.py`):

After extracting entities and relations, this script aggregates all samples (JSON Lines (jsonl) in a file) into a single aggregated graph.

For each input file:
a. Processes each line as a GoldLabelDataset object.
b. Extracts and aggregates unique entities, relations, and edges across all samples.
c. Normalizes the data by converting entities and predicates to lowercase.

The aggregated data is structured into an AggregatedDataset object containing:
- A sorted list of unique entities
- A sorted list of unique relations (as lists of [subject, predicate, object])
- A sorted list of unique edges (predicates)

The aggregated dataset is then saved as a JSON file for each input file, with the same name but a .json extension.

This aggregation step aggregates and reduces redundancy of exact entity and relationmatches.

### 3. Entity and Edge Clustering (`cluster.py`):
This script further refines the aggregated data from `aggregate.py` by clustering similar entities and edges.

It uses OpenAI's Structured Output feature with GPT-4o-2024-08-06 to dynamically group entities and edges based on semantic similarity.

The clustering process:
a. Iteratively extracts clusters of related items (entities or edges) using `extract_single_cluster()`.
b. Validates each cluster to ensure semantic coherence with `validate_cluster()`. This queries the cluster with GPT-4o to check if the cluster fulfills the criteria of being a coherent group.
c. Chooses a representative item for each cluster using `choose_representative()`.

Clustering helps to:
- Reducing redundancy by grouping semantically equivalent or closely related entities/edges.
- Handling variations in tenses, plural forms, stem forms, and capitalization.

Note that we were relatively conservative in our clustering process, and did not allow for many variations in entity/edge names.

The clustering process consists of two key parts:

1. Given the entire list of items (entities or edges), we extract a single cluster at a time using the `extract_single_cluster()` function. This process repeats until no more clusters can be extracted with this method. The extraction uses the following prompt:

For entities:
```
EXTRACT_ENTITIES_CLUSTER_PROMPT = """Find ONE cluster of related entities from this list.
A cluster should contain entities that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Or entities with close semantic meanings.

Return only if you find entities that clearly belong together.
If you can't find a clear cluster, return an empty list."""
```

For edges:
```
EXTRACT_EDGES_CLUSTER_PROMPT = """Find ONE cluster of closely related predicates from this list.
A cluster should contain predicates that are the same in meaning, with different:
- tenses
- plural forms
- stem forms
- upper/lower cases
Predicates are the relations between subject and object entities. Ensure that the predicates in the same cluster have very close semantic meanings to describe the relation between the same subject and object entities.

Return only if you find predicates that clearly belong together.
If you can't find a clear cluster, return an empty list."""
```

The extracted cluster is validated using a `ClusterResponse` class:

```python
class ClusterResponse(BaseModel):
    cluster: List[str]
    @model_validator(mode='after')
    def validate_entities_exist(self) -> 'ClusterResponse':
        if not hasattr(self, '_valid_entities'):
            return self
        self.cluster = [e for e in self.cluster if e in self._valid_entities]
        return self
```

2. With remaining items, we batch them in sets (using 10 at a time) and, given all clusters in context, decide whether to add the items to an existing cluster or create new single-item clusters. This is done using the `check_items_for_existing_clusters()` function.

We use the following structured output class:

```python
class ClusterIndexResponse(BaseModel):
    cluster_indices: List[Optional[int]] = Field(description="A list of indices corresponding to the clusters each item can be added to, or None if they don't fit any existing cluster")
```

The clustered data is saved as a JSON file, with separate lists for entity clusters and edge clusters, each sorted by cluster size. The script also outputs statistics for both entity and edge clusters, including minimum, maximum, and average cluster sizes, as well as counts of single-item and multi-item clusters.
