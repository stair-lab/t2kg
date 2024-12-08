import csv
import json

# Function to read CSV file and extract data
def read_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Read the CSV files
input_data = {}
csv_files = [
    'pku-safe-30k-gemma-2-9b_no_ann.csv',
    'pku-safe-30k-test-gemma-2-9b-it.csv',
    'pku-safe-30k-test-Mistral-7B-Instruct-v0.2.csv',
    'pku-safe-30k-test-Mistral-7B-v0.2_no_ann.csv'
]
for file in csv_files:
    input_data[file] = read_csv(file)

# Prepare data for JSONL and write to separate files
for file, data in input_data.items():
    output_file = file.replace('.csv', '.jsonl')
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for row in data:
            item = {
                'human_query': row['prompt'],
                'assistant_response': row['response'],
                'source_file': file
            }
            json.dump(item, jsonl_file)
            jsonl_file.write('\n')

print("Conversion complete. Output saved to separate JSONL files.")
