"""
This file contains utility classes and functions for knowledge graph extraction and processing.

Key components:
1. OpenAI client setup and API key verification
2. Pydantic models for knowledge graph entities and relations
3. Functions for entity extraction using OpenAI's GPT-4o structured output model
4. Classes for representing synthetic data and gold label datasets

The code uses OpenAI's API to extract entities and relations from text, 
and provides data structures to represent and manipulate knowledge graph components.

Note: Ensure that the OPENAI_API_KEY environment variable is set before running this script.
"""

from pydantic import BaseModel
from openai import OpenAI
import json
from typing import List, Dict, Tuple
import logging
from enum import Enum
from pydantic import Field, model_validator
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()

MODEL = "gpt-4o-2024-08-06"

# Define global variables for paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATA_DIR = os.path.join(BASE_DIR, 'source_data')
KGS_DIR = os.path.join(BASE_DIR, 'kgs')

# Check if the API key is set in the environment variables
api_key = os.environ.get('OPENAI_API_KEY')

if api_key:
    print(f"OpenAI API Key: {api_key[:5]}...{api_key[-5:]}")
else:
    print("OpenAI API Key is not set in the environment variables.")

class KnowledgeGraphEntities(BaseModel):
    entities: List[str]

def get_entity_extraction_prompt() -> str:
    return "Extract key entities from the given text. Extracted entities are nouns, verbs, or adjectives, particularly regarding sentiment. This is for an extraction task, please be thorough and accurate to the reference text."

def get_additional_entity_extraction_prompt(exclude: List[str]) -> str:
    return f"Extract key entities from the given text. Extracted entities are nouns, verbs, or adjectives, particularly regarding sentiment. This is for an extraction task, please be thorough and accurate to the reference text. (Exclude entities or similar entities to the following: {', '.join(exclude)})"

def get_relation_extraction_prompt() -> str:
    return """Extract subject-predicate-object triples from the assistant message. A predicate (1-3 words) defines the relationship between the subject and object. Relationship may be fact or sentiment based on assistant's message. Subject and object are entities. Entities provided are from the assistant message and prior conversation history, though you may not need all of them. This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
""

def handle_openai_error(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"OpenAI API error: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API error after {max_retries} attempts: {str(e)}")
                    return ""  # Return an empty string as requested
    return wrapper

@handle_openai_error
def extract_entities(text: str, system_prompt: str = get_entity_extraction_prompt()) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format=KnowledgeGraphEntities,
        temperature=0
    )
    parsed = completion.choices[0].message.parsed
    return parsed.entities if parsed else []

def create_entity_enum(entities: List[str]) -> str:
    """Creates an Enum class definition from a list of entities"""
    if not entities:
        raise ValueError("Cannot create EntityEnum with empty entities list")
        
    # Clean entity names to be valid Python identifiers
    clean_entities = []
    seen_clean_names = set()  # Prevent duplicate enum names
    
    for entity in entities:
        # Replace spaces and special chars with underscore
        clean_name = ''.join(c if c.isalnum() else '_' for c in entity).upper()
        # Ensure it starts with a letter
        if clean_name[0].isdigit():
            clean_name = 'E_' + clean_name
        # Remove consecutive underscores
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')
        # Remove trailing underscores
        clean_name = clean_name.rstrip('_')
        
        # Handle duplicate clean names
        base_name = clean_name
        counter = 1
        while clean_name in seen_clean_names:
            clean_name = f"{base_name}_{counter}"
            counter += 1
        
        seen_clean_names.add(clean_name)
        clean_entities.append((entity, clean_name))

    enum_code = "class EntityEnum(str, Enum):\n"
    for original, clean in clean_entities:
        enum_code += f'    {clean} = "{original}"\n'
    
    return enum_code

def create_relation_models(entities: List[str]) -> Tuple[type, type]:
    """Creates KnowledgeGraphRelation and KnowledgeGraphRelations classes with entity validation"""
    
    # Create and execute EntityEnum
    enum_code = create_entity_enum(entities)
    namespace = {}
    exec(enum_code, globals(), namespace)
    EntityEnum = namespace['EntityEnum']
    
    # Log the created enum
    logger.debug(f"Created EntityEnum: {EntityEnum.__members__}")
    
    class KnowledgeGraphRelation(BaseModel):
        subject: EntityEnum = Field(..., description="Subject entity from the provided EntityEnum")
        predicate: str = Field(..., description="Predicate describing the relationship")
        object: EntityEnum = Field(..., description="Object entity from the provided EntityEnum")
        
    class KnowledgeGraphRelations(BaseModel):
        relations: List[KnowledgeGraphRelation]

        @model_validator(mode='after')
        def validate_no_self_relations(self) -> 'KnowledgeGraphRelations':
            for relation in self.relations:
                if relation.subject == relation.object:
                    raise ValueError(f"Self-relation detected: {relation}")
            return self
        
    # Log the created models
    logger.debug(f"Created KnowledgeGraphRelation model with fields: {KnowledgeGraphRelation.__fields__.keys()}")
    logger.debug(f"Created KnowledgeGraphRelations model with fields: {KnowledgeGraphRelations.__fields__.keys()}")
    
    return KnowledgeGraphRelation, KnowledgeGraphRelations

@handle_openai_error
def extract_relations(text: str, entities: List[str], system_prompt: str = get_relation_extraction_prompt()) -> List[Tuple[str, str, str]]:
    if not entities:
        logger.warning("No entities provided for relation extraction")
        return []
        
    try:
        # Create model classes with entity validation
        KnowledgeGraphRelation, KnowledgeGraphRelations = create_relation_models(entities)
        logger.debug(f"Created relation models with {len(entities)} valid entities")
        
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Message:\n {text}\n\n\nValid entities: {entities}"}
            ],
            response_format=KnowledgeGraphRelations,
            temperature=0
        )
        
        relations = completion.choices[0].message.parsed.relations
        # Convert EntityEnum values back to strings for the return value
        valid_relations = [(relation.subject.value, relation.predicate, relation.object.value) 
                         for relation in relations]
        logger.info(f"Successfully extracted {len(valid_relations)} valid relations")
        return valid_relations
        
    except ValueError as e:
        logger.error(f"Validation error in relation extraction: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in relation extraction: {e}")
        return []

class SyntheticData(BaseModel):
    query_entities: List[str] = Field(..., description="Entities extracted from the user query")
    entities_all: List[str] = Field(..., description="All entities including additional ones from response")
    relations: List[Tuple[str, str, str]] = Field(..., description="List of (subject, predicate, object) triples")
    data_generation_model: str = Field(..., description="Model used for data generation")

    @model_validator(mode='after')
    def validate_entities_in_relations(self) -> 'SyntheticData':
        all_entities = set(self.entities_all)
        for subj, _, obj in self.relations:
            if subj not in all_entities:
                raise ValueError(f"Relation subject '{subj}' not in entities_all")
            if obj not in all_entities:
                raise ValueError(f"Relation object '{obj}' not in entities_all")
        return self

class GoldLabelDataset(BaseModel):
    user_query: str
    synthetic: SyntheticData
    hh_rlhf_assistant_response: str

def process_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file if json.loads(line)]

@handle_openai_error
def extract_additional_entities(text: str, exclude: List[str], system_prompt: str = None) -> List[str]:
    if system_prompt is None:
        system_prompt = get_additional_entity_extraction_prompt(exclude)
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format=KnowledgeGraphEntities,
        temperature=0
    )
    return completion.choices[0].message.parsed.entities

def process_sample(user_query: str, assistant_response: str, shared_query_entities: List[str]) -> GoldLabelDataset:
    try:
        logger.info(f"Processing sample with {len(shared_query_entities)} query entities")
        
        additional_entities = extract_additional_entities(assistant_response, shared_query_entities)
        logger.info(f"Extracted {len(additional_entities)} additional entities: {additional_entities}")
        
        all_entities = shared_query_entities + additional_entities
        
        relations = extract_relations(assistant_response, all_entities)
        logger.info(f"Extracted {len(relations)} relations: {relations}")
        
        return GoldLabelDataset(
            user_query=user_query,
            synthetic=SyntheticData(
                query_entities=shared_query_entities,
                entities_all=all_entities,
                relations=relations,
                data_generation_model=MODEL,
            ),
            hh_rlhf_assistant_response=assistant_response
        )
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        # Return a valid but empty dataset
        return GoldLabelDataset(
            user_query=user_query,
            synthetic=SyntheticData(
                query_entities=shared_query_entities,
                entities_all=shared_query_entities,
                relations=[],
                data_generation_model=MODEL,
            ),
            hh_rlhf_assistant_response=assistant_response
        )

if __name__ == "__main__":    
    print(f"SOURCE_DATA_DIR: {SOURCE_DATA_DIR}")

    # List of files to process
    files_to_process = [
        # "harmless_base_rejected_test_50.jsonl"
        # "advil.jsonl"
        # "224w_final_deliverable/test.jsonl",
        "224w_final_deliverable/cleaned_300_pku-safe-30k-test-Mistral-7B-v0.2_no_ann.jsonl",
        "224w_final_deliverable/cleaned_300_pku-safe-30k-test-gemma-2-9b-it.jsonl",
        "224w_final_deliverable/cleaned_300_pku-safe-30k-test-Mistral-7B-Instruct-v0.2.jsonl",
        "224w_final_deliverable/cleaned_300_pku-safe-30k-gemma-2-9b_no_ann.jsonl"
    ]

    # Extract query entities from first file
    all_query_entities = set()
    query_entities_list = []
    file_path = os.path.join(SOURCE_DATA_DIR, files_to_process[0])
    print(f"Processing file: {file_path}")
    data = process_jsonl(file_path)
    print(f"Number of samples in {file_path}: {len(data)}")
    for sample in data:
        user_query = sample['human_query']
        print(f"Processing user query: {user_query}")
        extracted_entities = extract_entities(user_query)
        print(f"Extracted entities: {extracted_entities}")
        all_query_entities.update(extracted_entities)
        query_entities_list.append(extracted_entities)
    print(f"Total unique entities extracted: {len(all_query_entities)}")
    shared_query_entities = list(all_query_entities)
    print(f"Shared query entities: {shared_query_entities}")
    print(f"Query-specific entities: {query_entities_list}")
    # Save extracted entities and query-specific entities for future reference
    entities_output_file = os.path.join(KGS_DIR, "extracted_entities.json")
    with open(entities_output_file, 'w') as f:
        json.dump({
            "shared_query_entities": list(shared_query_entities),
            "query_specific_entities": query_entities_list
        }, f, indent=2)
    print(f"Saved extracted entities to {entities_output_file}")

    for file_name in files_to_process:
        file_path = os.path.join(SOURCE_DATA_DIR, file_name)
        print(f"Processing {file_path}...")
        
        data = process_jsonl(file_path)
        gold_label_dataset = []
        # Open the output file for writing
        output_file = os.path.join(KGS_DIR, file_name)
        with open(output_file, 'w') as f:
            for i, sample in enumerate(data):
                user_query = sample['human_query']
                assistant_response = sample['assistant_response']
                sample_query_entities = query_entities_list[i] if i < len(query_entities_list)  else []
                processed_sample = process_sample(user_query, assistant_response, sample_query_entities)
                gold_label_dataset.append(processed_sample)
                
                # Write the processed sample to the file immediately
                f.write(json.dumps(processed_sample.model_dump()) + '\n')

        print(f"Processed {len(gold_label_dataset)} samples from {file_name}.")
        print(f"Saved results to {output_file}")
        print()  # Add a blank line for readability between files
