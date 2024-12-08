import json
import random
import os
import argparse

def subsample_jsonl(input_files, n, output_dir):
    # Read all samples from the first file to get total count
    with open(input_files[0], 'r') as f:
        all_samples = [json.loads(line) for line in f]
    
    total_samples = len(all_samples)
    
    # If n is greater than total samples, use all samples
    if n > total_samples:
        n = total_samples
        print(f"Warning: n is greater than total samples. Using all {n} samples.")
    
    # Generate random indices for subsampling
    subsample_indices = random.sample(range(total_samples), n)
    
    # Process each input file
    for input_file in input_files:
        # Read all samples
        with open(input_file, 'r') as f:
            all_samples = [json.loads(line) for line in f]
        
        # Subsample using the same indices
        subsampled = [all_samples[i] for i in subsample_indices]
        
        # Create output filename
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{n}_{base_name}")
        
        # Write subsampled data to output file
        with open(output_file, 'w') as f:
            for sample in subsampled:
                json.dump(sample, f)
                f.write('\n')
        
        print(f"Subsampled {n} samples from {input_file} to {output_file}")

if __name__ == "__main__":
    input_files = [
        "pku-safe-30k-gemma-2-9b_no_ann.jsonl",
        "pku-safe-30k-test-gemma-2-9b-it.jsonl",
        "pku-safe-30k-test-Mistral-7B-Instruct-v0.2.jsonl",
        "pku-safe-30k-test-Mistral-7B-v0.2_no_ann.jsonl"
    ]
    n = 300
    output_dir = "."
    
    subsample_jsonl(input_files, n, output_dir)
