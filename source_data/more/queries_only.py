import json

def process_dialogue(dialogue):
    parts = dialogue.strip().split("\n\n")
    if len(parts) < 2:
        return "", ""  # Return empty strings if there aren't enough parts
    
    human_query = parts[0].replace("Human: ", "")
    assistant_response = parts[1].replace("Assistant: ", "") if len(parts) > 1 else ""
    
    return human_query, assistant_response

def process_jsonl(file_path, output_chosen, output_rejected):
    with open(file_path, 'r') as file, \
         open(output_chosen, 'w') as chosen_file, \
         open(output_rejected, 'w') as rejected_file:
        
        for line in file:
            item = json.loads(line)
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')
            
            chosen_query, chosen_response = process_dialogue(chosen)
            rejected_query, rejected_response = process_dialogue(rejected)
            
            json.dump({"human_query": chosen_query, "assistant_response": chosen_response}, chosen_file)
            chosen_file.write('\n')
            
            json.dump({"human_query": rejected_query, "assistant_response": rejected_response}, rejected_file)
            rejected_file.write('\n')

# Process test.jsonl
test_file_path = 'test.jsonl'
process_jsonl(test_file_path, 'chosen_test.jsonl', 'rejected_test.jsonl')

# Process train.jsonl
train_file_path = 'train.jsonl'
process_jsonl(train_file_path, 'chosen_train.jsonl', 'rejected_train.jsonl')

print("Processing complete. Output files created: chosen_test.jsonl, rejected_test.jsonl, chosen_train.jsonl, rejected_train.jsonl")
