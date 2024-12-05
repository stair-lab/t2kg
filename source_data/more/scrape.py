
import json

def count_dialogue_turns(text):
    return text.count("\n\nHuman:") + text.count("\n\nAssistant:")

def process_jsonl(file_path):
    valid_items = []
    
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')
            
            if count_dialogue_turns(chosen) == 2 and count_dialogue_turns(rejected) == 2:
                valid_items.append(item)
    
    return valid_items
# Process test.jsonl
test_file_path = 'harmless-base/test.jsonl'
test_result = process_jsonl(test_file_path)

# Process train.jsonl
train_file_path = 'harmless-base/train.jsonl'
train_result = process_jsonl(train_file_path)

# Output test results to ./test.jsonl
with open('./test.jsonl', 'w') as outfile:
    for item in test_result:
        json.dump(item, outfile)
        outfile.write('\n')

# Output train results to ./train.jsonl
with open('./train.jsonl', 'w') as outfile:
    for item in train_result:
        json.dump(item, outfile)
        outfile.write('\n')

# Print summary
print(f"Number of valid items in test set: {len(test_result)}")
print(f"Number of valid items in train set: {len(train_result)}")
