import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

os.environ["WANDB_PROJECT"] = "grpo"
os.environ["WANDB_LOG_MODEL"] = "baseline"

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
import torch

from rewards import (
    xmlcount_reward_func,
    correctness_reward_func_orig,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
)
from datasets import dataset, eval_dataset
from models import context_model as model, context_tokenizer as tokenizer

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 1536,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "baseline",
    # eval_strategy = "steps",
    # eval_steps = 100,
    # eval_on_start = True,
    # per_device_eval_batch_size = 1,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func_orig,
    ],
    args = training_args,
    train_dataset = dataset,
    eval_dataset=eval_dataset,
)
trainer.train()