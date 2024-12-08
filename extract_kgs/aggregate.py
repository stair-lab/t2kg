"""
This script aggregates data from multiple JSON Lines (jsonl) files containing knowledge graph information.
It processes each file, extracts entities, relations, and edges, and combines them into a single dataset.
The aggregated data is then saved as a JSON file for each input file.
This aggregation step is crucial for creating a comprehensive knowledge graph from multiple sources.
"""

from typing import List, Dict, Set, Tuple
import json
from pydantic import BaseModel
from index import GoldLabelDataset

class AggregatedDataset(BaseModel):
    entities: List[str]
    relations: List[List[str]]
    edges: List[str]

def aggregate_dataset(input_file: str) -> AggregatedDataset:
    aggregated_data: Dict[str, Set] = {
        "entities": set(),
        "relations": set(),
        "edges": set()
    }

    with open(f'kgs/{input_file}', 'r') as f:
        for line in f:
            dataset = GoldLabelDataset(**json.loads(line))
            aggregated_data["entities"].update(entity.lower() for entity in dataset.synthetic.entities_all)
            for s, p, o in dataset.synthetic.relations:
                relation = (s.lower(), p.lower(), o.lower())
                aggregated_data["relations"].add(relation)
                aggregated_data["edges"].add(p.lower())

    return AggregatedDataset(
        entities=sorted(list(aggregated_data["entities"])),
        relations=sorted([list(rel) for rel in aggregated_data["relations"]]),
        edges=sorted(list(aggregated_data["edges"]))
    )

if __name__ == "__main__":
    input_files = [
        # 'advil.jsonl'
        # 'harmless_base_rejected_test_50.jsonl',
        # 'harmless_base_rejected_test_50.jsonl',
        # 'llama_policy.jsonl',
        # 'llama_reference.jsonl',
        # 'pythia28_policy.jsonl',
        # 'pythia28_reference.jsonl'
        # '224w_final_deliverable/cleaned_300_pku-safe-30k-Mistral-7B-Instruct-v0.2.jsonl',
        '224w_final_deliverable/cleaned_300_pku-safe-30k-test-gemma-2-9b-it.jsonl',
        '224w_final_deliverable/cleaned_300_pku-safe-30k-test-Mistral-7B-v0.2_no_ann.jsonl',
    ]
    
    for input_file in input_files:
        aggregated_result = aggregate_dataset(input_file)
        output_file = f'kgs/{input_file.replace(".jsonl", ".json")}'
        
        with open(output_file, 'w') as f:
            json.dump(aggregated_result.dict(), f, indent=2)
        
        print(f"Aggregated data for {input_file} saved to {output_file}")