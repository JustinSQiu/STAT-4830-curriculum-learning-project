import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

os.environ["WANDB_PROJECT"] = "grpo"
os.environ["WANDB_LOG_MODEL"] = "llama"

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from vllm import SamplingParams

max_seq_length = 1024
lora_rank = 32

# Load the base model and tokenizer
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,        # using 4-bit inference in this example
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.3,
)
# Freeze the base model parameters if needed
for param in base_model.parameters():
    param.requires_grad = False

# Load the context model and tokenizer
context_model, context_tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.3,
)

# Wrap the context model with PEFT (LoRA) modifications as was done during training
context_model = FastLanguageModel.get_peft_model(
    context_model,
    r=lora_rank,   
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

os.chdir('/mnt/castor/seas_home/j/jsq/dev/STAT-4830-curriculum-learning-project/reasoning_context_model/')
context_model.load_lora("grpo_saved_lora")

# ----- Inference Pipeline -----

# Define your input question.
question = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

# Step 1: Generate a context using the context model.
# Here you can design your prompt for context generation as needed.
# For example, you might simply pass the question along with an instruction to generate supportive context.
context_prompt = f"Given the math question below, write a step-by-step reasoning context that clearly explains concepts needed to solve the problem\nQuestion: {question}"

# Set up sampling parameters for context generation.
sampling_params_context = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,  # adjust based on expected context length
)

# Generate the context using the context model.
context_output = context_model.fast_generate(
    context_prompt,
    sampling_params=sampling_params_context,
    lora_request = context_model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text.strip()

print("Generated context:")
print(context_output)
print("\n------------------\n")

# Step 2: Generate an answer using the base model.
# Create a prompt that includes the original question and the generated context.
answer_prompt = f"Question: {question}\nContext: {context_output}\nThe answer is:"

# Set up sampling parameters for answer generation.
sampling_params_answer = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,  # adjust based on expected answer length
)

# Generate the answer using the base model.
answer_output = base_model.fast_generate(
    answer_prompt,
    sampling_params=sampling_params_answer,
)[0].outputs[0].text.strip()

print("Generated answer:")
print(answer_output)
