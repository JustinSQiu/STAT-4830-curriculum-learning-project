import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch
import re
from datasets import load_dataset, Dataset
from unsloth import is_bfloat16_supported

# Hyperparameters
max_seq_length = 512
lora_rank = 32

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)
# Freeze all parameters for K
for param in base_model.parameters():
    param.requires_grad = False

question = "What is the capital of France?"
context = "You should return just the word Paris."
ans = "Paris"

k_input = f"Question: {question}\nContext: {context}\nAnswer:"

# Tokenize inputs
input_ids = base_tokenizer.encode(k_input, return_tensors='pt').to(base_model.device)
# Encode the ground truth answer
answer_ids = base_tokenizer.encode(ans, return_tensors='pt').to(base_model.device)
# Create full input: conditioning text + answer (for computing likelihood)
full_input = base_tokenizer.encode(k_input + ans, return_tensors='pt').to(base_model.device)

with torch.no_grad():
    outputs = base_model(full_input)
logits = outputs.logits  # shape: [1, sequence_length, vocab_size]

# The answer tokens start after the input_ids
start_idx = input_ids.shape[1]
# For causal LM: logits[i] gives the distribution for token at position i.
# We extract logits corresponding to the answer tokens.
answer_logits = logits[0, start_idx-1:-1, :]
target_ids = full_input[0, start_idx:]
# Compute log probabilities
log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
# Gather log-probabilities of the target answer tokens
token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
total_log_prob = token_log_probs.sum().item()
# Normalize reward by the number of tokens (to maintain consistent scale)
reward = total_log_prob / target_ids.shape[0]
print(f"Reward: {reward}")