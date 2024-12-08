import json
import os

def remove_human_assistant_tags(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            human_query = data['human_query']

            # Remove "Human: " from the beginning and "\nAssistant:" from the end
            human_query = human_query.removeprefix("Human: ").removesuffix("\nAssistant:")

            # Update the human_query in the data
            data['human_query'] = human_query

            # Write the updated data to the output file
            json.dump(data, f_out)
            f_out.write('\n')

def process_files():
    files_to_process = [
        "processed_300_pku-safe-30k-test-gemma-2-9b-it.jsonl",
        "processed_300_pku-safe-30k-test-Mistral-7B-Instruct-v0.2.jsonl"
    ]

    for file_name in files_to_process:
        input_file = file_name
        output_file = f"cleaned_{file_name}"
        remove_human_assistant_tags(input_file, output_file)
        print(f"Processed {file_name} and saved as {output_file}")

if __name__ == "__main__":
    process_files()
