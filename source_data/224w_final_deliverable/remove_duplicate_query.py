import json
import os

def remove_duplicate_query(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            human_query = data['human_query']
            assistant_response = data['assistant_response']

            # Remove the human query from the beginning of the assistant response
            if assistant_response.startswith(human_query):
                assistant_response = assistant_response[len(human_query):].lstrip()

            # Update the assistant_response in the data
            data['assistant_response'] = assistant_response

            # Write the updated data to the output file
            json.dump(data, f_out)
            f_out.write('\n')

def process_files():
    files_to_process = [
        # "300_pku-safe-30k-gemma-2-9b_no_ann.jsonl",
        # "300_pku-safe-30k-test-Mistral-7B-v0.2_no_ann.jsonl"
        "300_pku-safe-30k-test-Mistral-7B-Instruct-v0.2.jsonl",
        "300_pku-safe-30k-test-gemma-2-9b-it.jsonl"
    ]

    for file_name in files_to_process:
        input_file = file_name
        output_file = f"processed_{file_name}"
        remove_duplicate_query(input_file, output_file)
        print(f"Processed {file_name} and saved as {output_file}")

if __name__ == "__main__":
    process_files()
