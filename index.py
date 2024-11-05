from pydantic import BaseModel
from openai import OpenAI
import json
from typing import List, Dict, Tuple

client = OpenAI()

MODEL = "gpt-4o-2024-08-06"
SOURCE_FILE = './source_data/test_small.jsonl'

class KnowledgeGraphEntities(BaseModel):
    entities: List[str]


def get_entity_extraction_prompt() -> str:
    return "Extract entities from the given text."


def extract_entities(text: str, system_prompt: str = get_entity_extraction_prompt()) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format=KnowledgeGraphEntities,
    )
    parsed = completion.choices[0].message.parsed
    return parsed.entities if parsed else []




class KnowledgeGraphRelation(BaseModel):
    subject: str
    predicate: str
    object: str

class KnowledgeGraphRelations(BaseModel):
    relations: List[KnowledgeGraphRelation]

class SyntheticData(BaseModel):
    query_entities: List[str]
    entities_all: List[str]
    relations: List[Tuple[str, str, str]]
    data_generation_model: str
    # is_consistent: bool

class GoldLabelDataset(BaseModel):
    user_query: str
    synthetic: SyntheticData
    hh_rlhf_assistant_response: str

def process_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file if json.loads(line)]
    
def get_entity_extraction_prompt() -> str:
    return "Extract entities from the given text."

def get_additional_entity_extraction_prompt(exclude: List[str]) -> str:
    return f"Extract entities from the given text. (Exclude entities or similar entities to the following: {', '.join(exclude)})"

def get_relation_extraction_prompt() -> str:
    return "Extract predicates between entities as subject-predicate-object triples. Entities must be subject or object."

def extract_entities(text: str, system_prompt: str = get_entity_extraction_prompt()) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format=KnowledgeGraphEntities,
    )
    parsed = completion.choices[0].message.parsed
    return parsed.entities if parsed else []

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
    )
    return completion.choices[0].message.parsed.entities

def extract_relations(text: str, entities: List[str], system_prompt: str = get_relation_extraction_prompt()) -> List[Tuple[str, str, str]]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: {text}\nEntities: {entities}"},
        ],
        response_format=KnowledgeGraphRelations,
    )
    
    relations = completion.choices[0].message.parsed.relations
    return [(relation.subject, relation.predicate, relation.object) for relation in relations]

# class IsConsistent(BaseModel):
#     answer: bool
#     reasoning: str

# # Perform consistency check
# def check_consistency(relations: List[Tuple[str, str, str]], text: str) -> bool:
#     completion = client.beta.chat.completions.parse(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": "Verify that these subject-predicate-object triples represent a comprehensive knowledge graph of the following text. Explain your reasoning."},
#             {"role": "user", "content": f"Text: {text}\nExtracted triples: {relations}\nAre these relations accurate and consistent with the text?"}
#         ],
#         response_format=IsConsistent
#     )
#     parsed_response = completion.choices[0].message.parsed
#     return parsed_response.answer, parsed_response.reasoning

def process_sample(user_query: str, assistant_response: str) -> GoldLabelDataset:
    query_entities = extract_entities(user_query)
    print(f"Query entities: {query_entities}")
    
    additional_entities = extract_additional_entities(assistant_response, query_entities)
    print(f"Additional entities: {additional_entities}")
    
    all_entities = query_entities + additional_entities
    print(f"All entities: {all_entities}")

    relations = extract_relations(assistant_response, all_entities)
    print(f"Relations: {relations}")
    
    # is_consistent = False
    # attempts = 0
    # max_attempts = 4

    # while not is_consistent and attempts < max_attempts:
    #     is_consistent, reasoning = check_consistency(relations, assistant_response)
    #     print(f"Consistency check attempt {attempts + 1} passed: {is_consistent}")
    #     print(f"Reasoning: {reasoning}")
    #     if not is_consistent:
    #         print(f"Warning: Extracted relations may not be consistent with the text. Attempt {attempts + 1} of {max_attempts}.")
    #         if attempts < max_attempts - 1:
    #             print("Retrying entity extraction and relation extraction...")
    #             updated_entities_prompt = f"Given the text, revise this list of entities: {all_entities}\nWhat is wrong with the triples: {reasoning}"
    #             # . Please (1) keep relevant entities (2) add new entities that were missed before (3) remove irrelevant entities.
    #             all_entities = extract_additional_entities(assistant_response, [], updated_entities_prompt)
    #             print(f"New all entities: {all_entities}")
                
    #             updated_relation_prompt = f"Extract predicates between entities as subject-predicate-object triples. Entities must be subject or object. Previously extracted relations: {relations}. Please revise this list of subject-predicate-object triples. What is wrong with the triples: {reasoning}"
    #             # Please (1) keep relevant relations (2) refine existing relations (3) add new relevant relations (4) remove irrelevant relations, ensuring all entities are properly connected.
    #             relations = extract_relations(assistant_response, all_entities, updated_relation_prompt)
    #             print(f"New relations: {relations}")
    #     attempts += 1

    # if not is_consistent:
    #     print(f"Warning: After {max_attempts} attempts, extracted relations are still not consistent with the text.")
    
    return GoldLabelDataset(
        user_query=user_query,
        synthetic=SyntheticData(
            query_entities=query_entities or [],
            entities_all=all_entities or [],
            relations=relations or [],
            data_generation_model=MODEL,
            # is_consistent=is_consistent
        ),
        hh_rlhf_assistant_response=assistant_response
    )

if __name__ == "__main__":
    # file_path = 'source_data/test_small.jsonl'
    
    # List of files to process
    files_to_process = [
        # "test_small.jsonl"
        'llama_policy.jsonl',
        'llama_reference.jsonl',
        'pythia28_policy.jsonl',
        'pythia28_reference.jsonl'
    ]

    for file_name in files_to_process:
        file_path = f'source_data/{file_name}'
        print(f"Processing {file_path}...")
        
        data = process_jsonl(file_path)
        gold_label_dataset = []

        for sample in data:
            user_query = sample['original_query']
            assistant_response = sample['assistant_response']
            processed_sample = process_sample(user_query, assistant_response)
            gold_label_dataset.append(processed_sample)

        # Save the processed data
        output_file = f'./extracted_graphs/{file_name}'
        with open(output_file, 'w') as f:
            for item in gold_label_dataset:
                f.write(json.dumps(item.model_dump()) + '\n')

        print(f"Processed {len(gold_label_dataset)} samples from {file_name}.")
        print(f"Saved results to {output_file}")
        print()  # Add a blank line for readability between files
    # file_path = SOURCE_FILE
    # data = process_jsonl(file_path)
    # gold_label_dataset = []

    # for sample in data:
    #     user_query = sample['chosen'].split('\n\nHuman: ')[1].split('\n\nAssistant: ')[0]
    #     assistant_response = sample['chosen'].split('\n\nAssistant: ')[1]
    #     processed_sample = process_sample(user_query, assistant_response)
    #     gold_label_dataset.append(processed_sample)

    # # Save the processed data
    # with open('./extracted/gold_label_dataset.jsonl', 'w') as f:
    #     for item in gold_label_dataset:
    #         f.write(json.dumps(item.model_dump()) + '\n')

    # print(f"Processed {len(gold_label_dataset)} samples.")
