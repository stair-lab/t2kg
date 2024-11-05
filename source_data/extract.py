import csv
import json

def process_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for row in reader:
                prompt = row['prompt']
                sample = row['sample']
                # Find the last occurrence of "Human:" followed by "Assistant:"
                last_human_index = prompt.rfind("Human:")
                last_assistant_index = prompt.rfind("Assistant:", last_human_index)
                
                if last_human_index != -1 and last_assistant_index != -1:
                    # Extract the last human query
                    last_human_query = prompt[last_human_index:last_assistant_index].strip()
                    # Remove "\n\nAssistant:" from the end of last_human_query if present
                    
                    # Find and remove the last human query and everything before it from the sample
                    assistant_response_start = sample.rfind(last_human_query)
                    if assistant_response_start != -1:
                        assistant_response = sample[assistant_response_start + len(last_human_query):].strip()
                        # Remove the "Assistant:" prefix if present
                        # assistant_response = assistant_response.lstrip("Assistant:").strip()
                    else:
                        assistant_response = sample  # Fallback if the query is not found in the sample
                else:
                    assistant_response = sample  # Fallback if the expected format is not found
                output = {
                    "original_query": prompt.rstrip("\n\nAssistant:"),
                    "assistant_response": assistant_response
                }
                
                json.dump(output, jsonl_file, ensure_ascii=False)
                jsonl_file.write('\n')
# Process all CSV files in the current folder
csv_files = ['llama_policy.csv', 'llama_reference.csv', 'pythia28_policy.csv', 'pythia28_reference.csv']

for input_csv in csv_files:
    output_jsonl = input_csv.replace('.csv', '.jsonl')
    process_csv(input_csv, output_jsonl)
    print(f"Processed {input_csv} to {output_jsonl}")

