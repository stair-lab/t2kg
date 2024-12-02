
from typing import List, Dict
import json
from index import GoldLabelDataset

def aggregate_dataset(input_file: str) -> Dict:
    aggregated_data = {
        "entities": set(),
        "relations": set()
    }

    with open(f'kgs/exp2/{input_file}', 'r') as f:
        for line in f:
            dataset = GoldLabelDataset(**json.loads(line))
            aggregated_data["entities"].update(entity.lower() for entity in dataset.synthetic.entities_all)
            aggregated_data["relations"].update((s.lower(), p.lower(), o.lower()) for s, p, o in dataset.synthetic.relations)

    return {
        "entities": sorted(list(aggregated_data["entities"])),
        "relations": sorted(list(map(list, aggregated_data["relations"])))
    }

if __name__ == "__main__":
    input_files = [
        'harmless_base_chosen_test_50.jsonl',
        'harmless_base_rejected_test_50.jsonl',
        # 'llama_policy.jsonl',
        # 'llama_reference.jsonl',
        # 'pythia28_policy.jsonl',
        # 'pythia28_reference.jsonl'
    ]
    
    for input_file in input_files:
        aggregated_result = aggregate_dataset(input_file)
        output_file = f'kgs/exp2/{input_file.replace(".jsonl", ".json")}'
        
        with open(output_file, 'w') as f:
            json.dump(aggregated_result, f, indent=2)
        
        print(f"Aggregated data for {input_file} saved to {output_file}")