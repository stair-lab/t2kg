from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from omegaconf import DictConfig
from datasets import Dataset, DatasetDict, load_dataset


total_samples = 100
model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "PKU-Alignment/PKU-SafeRLHF-30K"
max_length = 512
batch_size = 3
temperature = 0.7
top_k = 50
top_p = 0.95
DEVICE = 'cuda'
push_dataset_name = 'jkazdan/pku-safe-llama-3.1-8B-Instruct-test'

def prepare_data(dataset_name):
    if dataset_name == 'anthropic/hh-rlhf':
        dataset = load_dataset(dataset_name, split = "test")
        prompts = dataset['chosen']
        for i, prompt in enumerate(prompts):
            prompts[i] = prompt.split("Assistant:")[0] + "\nAssistant: "
    elif 'PKU-Alignment' in dataset_name:
        dataset = load_dataset(dataset_name, split = "test")
        prompts = dataset['prompt']
        for i, prompt in enumerate(prompts):
            prompts[i]= 'Human: ' + prompt + "\nAssistant:"
    else:
        assert NotImplementedError
    return prompts

def generate_prompts(model_name, prompts, total_samples, batch_size, max_length):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_out = []
    resp_out = []

    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for i in range(len(prompts)//batch_size):
        if i*batch_size >= total_samples:
            break
        batch = prompts[i*batch_size:(i+1)*batch_size]
        input_ids = tokenizer(batch, return_tensors="pt", padding = True).input_ids.to(DEVICE)
        # Generate text for the batch
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        prompt_out += batch
        # Decode and print each generated text in the batch
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            if dataset_name == 'anthropic/hh-rlhf':
                resp_out.append(generated_text.split("Assistant:")[1])
            else:
                resp_out.append(generated_text)
            print('##############')
        print(f'generated{len(resp_out)}')
    
    
    dataset = Dataset.from_dict(
        {
            "prompt": prompt_out,
            "response": resp_out,
        }
    )
    # Push the dataset to HuggingFace.
    print("Pushing the dataset to HuggingFace...")
    commit_info = dataset.push_to_hub(repo_id=push_dataset_name)


def main():
    prompts = prepare_data(dataset_name)
    generate_prompts(model_name, prompts, total_samples, batch_size, max_length)
    
if __name__ == "__main__":
    main()
