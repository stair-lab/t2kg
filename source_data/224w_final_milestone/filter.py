import json
import csv

# Function to read JSONL file and extract data
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Read the JSONL files
chosen_data = read_jsonl('harmless_base_chosen_test_50.jsonl')
rejected_data = read_jsonl('harmless_base_rejected_test_50.jsonl')

# Prepare data for CSV
csv_data = []
for chosen, rejected in zip(chosen_data, rejected_data):
    csv_data.append([
        chosen['human_query'],
        chosen['assistant_response'],
        rejected['assistant_response']
    ])

# Write to CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Human Query', 'Chosen Response', 'Rejected Response'])
    writer.writerows(csv_data)

print("Conversion complete. Output saved to 'output.csv'.")
