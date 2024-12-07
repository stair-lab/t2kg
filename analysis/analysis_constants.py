"""
Analysis_constants.py
-----------------------------------------------------------------
This file contains constants for the analysis code of the project.
Constants for the project. The details are in the analysis README 
"""
import os 

TEST_MODE = True


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
    KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_BASE_PATH, "kgs")
    
EMBEDDINGS_BASE_PATH = os.path.join(DATA_BASE_PATH, "embeddings")
EMBEDDING_PLOTS_BASE_PATH = os.path.join(DATA_BASE_PATH, "embedding_plots")

TRANSR = "TransR"

EMBEDDING_MODELS = [TRANSR]
DEFAULT_EMBEDDING_MODEL = TRANSR


JSON_ENTITIES_KEY = "entities"
JSON_RELATIONS_KEY = "relations"

# For graph data 
TRIPLES_KEY = "triples"
ENTITIES_KEY = "entities"
RELATIONS_KEY = "relations"
RELATION_MAPPING_KEY = "relation_mapping"
ENTITY_MAPPING_KEY = "entity_mapping"
















