from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

MODEL="gpt-4o-2024-08-06"
SOURCE_FILE='./source_data/test_small.jsonl'

import json
from typing import List, Dict, Tuple
class KnowledgeGraphEntities(BaseModel):
    entities: List[str]

class KnowledgeGraphRelation(BaseModel):
  subject: str
  predicate: str
  object: str

class KnowledgeGraphRelations(BaseModel):
    relations: List[KnowledgeGraphRelation]

class SyntheticData(BaseModel):
    query_entities: List[str]
    chosen_entities_all: List[str]
    rejected_entities_all: List[str]
    chosen_relations: List[Tuple[str, str, str]]
    rejected_relations: List[Tuple[str, str, str]]
    data_generation_model: str

class GoldLabelDataset(BaseModel):
    user_query: str
    synthetic: SyntheticData
    hh_rlhf_assistant_response: Dict[str, str]

def process_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file if json.loads(line)]

def extract_entities(text: str) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract entities from the given text."},
            {"role": "user", "content": text},
        ],
        response_format=KnowledgeGraphEntities,
    )
    parsed = completion.choices[0].message.parsed
    return parsed.entities if parsed else []

__all__ = [
    'KnowledgeGraphEntities',
    'KnowledgeGraphRelation',
    'KnowledgeGraphRelations',
    'SyntheticData',
    'GoldLabelDataset',
]


def extract_additional_entities(text: str, exclude: List[str]) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"Extract entities from the given text. (Exclude entities or similar entities to the following: {', '.join(exclude)})"},
            {"role": "user", "content": text},
        ],
        response_format=KnowledgeGraphEntities,
    )
    return completion.choices[0].message.parsed.entities


def extract_relations(text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract predicates between entities as subject-predicate-object triples. Entities must be subject or object. \n"},
            {"role": "user", "content": f"Text: {text}\nEntities: {entities}"},
        ],
        response_format=KnowledgeGraphRelations,
    )
    relations = completion.choices[0].message.parsed.relations
    return [(relation.subject, relation.predicate, relation.object) for relation in relations]

def process_sample(sample: dict) -> GoldLabelDataset:
    query_entities = extract_entities(sample['chosen'].split('\n\nHuman: ')[1].split('\n\nAssistant: ')[0])
    print(f"Query entities: {query_entities}")
    
    chosen_entities = extract_additional_entities(sample['chosen'].split('\n\nAssistant: ')[1], query_entities)
    print(f"Additional chosen entities: {chosen_entities}")
    rejected_entities = extract_additional_entities(sample['rejected'].split('\n\nAssistant: ')[1], query_entities)
    print(f"Additional rejected entities: {rejected_entities}")
    
    # Combine query_entities with chosen_entities and rejected_entities
    chosen_entities = query_entities + chosen_entities
    rejected_entities = query_entities + rejected_entities
    print(f"All chosen entities: {chosen_entities}")
    print(f"All rejected entities: {rejected_entities}")

    chosen_relations = extract_relations(sample['chosen'].split('\n\nAssistant: ')[1], chosen_entities)
    print(f"Chosen relations: {chosen_relations}")
    rejected_relations = extract_relations(sample['rejected'].split('\n\nAssistant: ')[1], rejected_entities)
    print(f"Rejected relations: {rejected_relations}")
    
    # Handle cases where entities or edges might be empty
    return GoldLabelDataset(
        user_query=sample['chosen'].split('\n\nHuman: ')[1].split('\n\nAssistant: ')[0],
        synthetic=SyntheticData(
            query_entities=query_entities or [],
            chosen_entities_all=chosen_entities or [],
            rejected_entities_all=rejected_entities or [],
            chosen_relations=chosen_relations or [],
            rejected_relations=rejected_relations or [],
            data_generation_model=MODEL
        ),
        hh_rlhf_assistant_response={
            "chosen": sample['chosen'].split('\n\nAssistant: ')[1],
            "rejected": sample['rejected'].split('\n\nAssistant: ')[1]
        }
    )

# ~
# ~~
# ~~~ Main execution ~~~
# ~~
# ~
file_path = SOURCE_FILE  # Adjust path as needed
data = process_jsonl(file_path)
gold_label_dataset = [process_sample(sample) for sample in data]

# Save the processed data
with open('./extracted/gold_label_dataset.jsonl', 'w') as f:
    for item in gold_label_dataset:
        f.write(json.dumps(item.model_dump()) + '\n')

print(f"Processed {len(gold_label_dataset)} samples.")
