import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # Replace with your LLaMA model path on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
model.eval()  # Set model to evaluation mode
# Define padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
# Load the prompts dataset
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K")  # Replace with your dataset path
prompts = dataset["test"]["prompt"]
for i, prompt in enumerate(prompts):
    prompts[i] = "Human: "+prompt + "\nAssistant: "

# Batch processing settings
batch_size = 8  # Adjust batch size as needed for memory constraints

# Function to process prompts in batches and gather responses
def generate_responses_batched(prompts, batch_size):
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"].to('cuda'), max_length=512)
        # Decode responses
        decoded_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        responses.extend(decoded_responses)

    return responses

# Generate responses
responses = generate_responses_batched(prompts, batch_size)

# Create a new dataset with prompts and responses
# new_data = {"prompts": prompts, "responses": responses}
new_data = {"responses": responses}
new_dataset = Dataset.from_dict(new_data)

# Save the dataset locally
dataset_path = "jkazdan/pku-safe-llama-3.2-3B"
new_dataset.save_to_disk(dataset_path)

# Load the new dataset as DatasetDict for uploading
new_dataset_dict = DatasetDict({"train": new_dataset})

# Push the dataset to Hugging Face Hub
huggingface_repo = dataset_path  # Replace 'username' and dataset name
new_dataset_dict.push_to_hub(dataset_path)

print(f"Dataset uploaded to Hugging Face Hub at: https://huggingface.co/datasets/{huggingface_repo}")
