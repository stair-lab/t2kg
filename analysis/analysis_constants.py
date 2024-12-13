"""
Analysis_constants.py
-----------------------------------------------------------------
This file contains constants for the analysis code of the project.
Constants for the project. The details are in the analysis README 
"""
import os 

TEST_MODE = False

PROJECT_BASE_PATH = "/Users/proudmpala/Documents/Stanford/Classes/5/5.1/CS224W/cs224w-final"
ANALYSIS_BASE_PATH = os.path.join(PROJECT_BASE_PATH, "analysis")

ANALYSIS_SRC_PATH = os.path.join(ANALYSIS_BASE_PATH, "src")
UTILS_PATH = os.path.join(ANALYSIS_SRC_PATH, "utils")

TEST_DIR = os.path.join(ANALYSIS_BASE_PATH, "test_dir")
TEST_KG_BASE_PATH = os.path.join(TEST_DIR, "test_kgs")
TEST_DATA_BASE_PATH = os.path.join(TEST_DIR, "test_data")

if TEST_MODE:
    DATA_BASE_PATH = TEST_DATA_BASE_PATH
    KNOWLEDGE_BASE_PATH = TEST_KG_BASE_PATH

else: 
    DATA_BASE_PATH = os.path.join(ANALYSIS_BASE_PATH, "data")
    KNOWLEDGE_BASE_PATH = os.path.join(ANALYSIS_BASE_PATH, "kgs")
    
EMBEDDINGS_BASE_PATH = os.path.join(DATA_BASE_PATH, "embeddings")
EMBEDDING_PLOTS_BASE_PATH = os.path.join(DATA_BASE_PATH, "embedding_plots")
EMBEDDING_DIM = 64

IMPUTATION_BASE_PATH = os.path.join(DATA_BASE_PATH, "imputation")
ORIGINAL_TRIPLES_KEY = "original_triples"
PREDICTED_TRIPLES_KEY = "predicted_triples"
K_NEW_TRIPLES = 100

TRANSR = "TransR"
EMBEDDING_MODELS = [TRANSR]
DEFAULT_EMBEDDING_MODEL = TRANSR

JSON_ENTITIES_KEY = "entities"
JSON_RELATIONS_KEY = "relations"

TRIPLES_KEY = "triples"
ENTITIES_KEY = "entities"
RELATIONS_KEY = "relations"
RELATION_MAPPING_KEY = "relation_mapping"
ENTITY_MAPPING_KEY = "entity_mapping"
ENTITY_CLUSTERS_KEY = "entity_clusters"

QUERY_BASE_PATH = os.path.join(DATA_BASE_PATH, "query")

PINECONE = "pinecone"
QUERY_MODELS = [PINECONE]
DEFAULT_QUERY_MODEL = PINECONE

QUERY_SYSTEM_EXIT_COMMAND = "exit"
QUERY_QUERIES_KEY = "queries"
QUERY_RESPONSE_KEY = "responses"
K_TOP_RESPONSE_ENTITIES = 10

QUERY_SYSTEM_MODES = ["batch", "interactive"]
DEFAULT_QUERY_SYSTEM_MODE = "interactive"


CLUSTER_OUTPUT_BASE_PATH = os.path.join(DATA_BASE_PATH, "cluster")

PINE_API_KEY = 'pcsk_6ktnFZ_HVs4214sfzmCDpKVvqAA1YDy5Fz8kjL9AX9ScrFGxZ7D16ATqdqHAyAsJnv2CwK'


CLUSTER_TOXICITY_BASE_PATH = os.path.join(DATA_BASE_PATH, "cluster_toxicity")

AGGREGATE_TOXICITY_STATS_KEY = "aggregate_toxicity_stats"
CLUSTER_TOXICITY_STATS_KEY = "clusters_toxicity"

NEW_TRIPLES_ANALYSIS_BASE_PATH = os.path.join(DATA_BASE_PATH, "new_triples_toxicity")
NEW_TRIPLES_TOXICITY_STATS_KEY = "new_triples_toxicity"

CLUSTER_METRICS_BASE_PATH = os.path.join(DATA_BASE_PATH, "cluster_metrics")

CLUSTER_STATS_KEY_MEAN_SIZE = "mean_size"
CLUSTER_STATS_KEY_STD_SIZE = "std_size"
CLUSTER_STATS_KEY_MAX_SIZE = "max_size"
CLUSTER_STATS_KEY_MIN_SIZE = "min_size"















