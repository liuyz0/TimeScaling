import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import accelerate

custom_cache_dir = './cache' # cloud
batch_size = 12 # 48 documents as a batch
max_length = 1024 # max length to generate (should be <= model max length, 2048 for Pythia)
num_docs = batch_size * 400 # total number of documents to generate 400!
steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 6000, 12000, 16000, 32000, 48000, 79000, 143000] # each model 24GB, total 480GB
revisions = ["step"+str(step) for step in steps]

# chunk_size_time = None
accumulated_loss = torch.zeros(len(revisions))
total_tokens = torch.zeros(len(revisions))
accumulated_logit_std = torch.zeros(len(revisions))
accumulated_logit_mean = torch.zeros(len(revisions))
skewness = torch.zeros(len(revisions))
skewness_extreme = torch.zeros(len(revisions))
logit_range = torch.zeros(len(revisions))

model_name = "EleutherAI/pythia-12b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

for rev_idx, rev in enumerate(revisions):
    data_iter = iter(dataset)

    if rev == "step143000":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=custom_cache_dir,
            torch_dtype = torch.float16,
            device_map="auto"
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=rev,
            cache_dir=custom_cache_dir,
            torch_dtype = torch.float16,
            device_map="auto"
        ).eval()

    start_time = time.time()

    for step in range(num_docs // batch_size):
        batch = [next(data_iter)['text'] for _ in range(batch_size)]
        inputs = tokenizer(batch, return_tensors="pt", padding="max_length",
            truncation=True,
            max_length=max_length) # ids and attention masks (batch_size, seq_len)
        labels = inputs.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100 # (batch_size, seq_len)
        
        valid_mask = (inputs['attention_mask'][:, 1:] > 0) & (inputs['attention_mask'][:, :-1] > 0) # (batch_size, seq_len-1)
        
        with torch.no_grad():
            out = model(
                **inputs.to(model.device),
                labels=labels.to(model.device)
            )

        valid_mask = valid_mask.to(out.device)
        logits = out.logits[:,:-1][valid_mask].to(torch.float32) # (num_valid_tokens, vocab_size)
        valid_labels = labels[:,1:].to(out.device)[valid_mask] # (num_valid_tokens,)
        # gather the logits corresponding to the valid labels
        correct_logits = logits.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1) # (num_valid_tokens,)
        total_tokens[rev_idx] += logits.shape[0]
        accumulated_logit_std[rev_idx] += logits.std(dim = -1).sum().item() / max_length
        accumulated_logit_mean[rev_idx] += logits.mean(dim = -1).sum().item() / max_length
        accumulated_loss[rev_idx] += out.loss.to(torch.float32).item() * logits.shape[0]
        skewness[rev_idx] += (correct_logits - logits.mean(dim = -1)).sum().item() / max_length
        # skeewness_extreme: find the maximum logits - mean logits
        skewness_extreme[rev_idx] += (logits.max(dim = -1).values - logits.mean(dim = -1)).sum().item() / max_length
        logit_range[rev_idx] += (logits.max(dim = -1).values - logits.min(dim = -1).values).sum().item() / max_length

    del model
    print(f"Revision {rev}, Time for {num_docs} docs: {time.time() - start_time:.2f} seconds")

torch.save({'loss': accumulated_loss / total_tokens, 'logit_std': accumulated_logit_std / total_tokens * max_length, 
            'logit_mean': accumulated_logit_mean / total_tokens * max_length, 'total_tokens': total_tokens,
            'skewness': skewness / total_tokens * max_length, 'skewness_extreme': skewness_extreme / total_tokens * max_length,
            'logit_range': logit_range / total_tokens * max_length},
           f'../outputs/pythia-logit-1.pt') # cloud