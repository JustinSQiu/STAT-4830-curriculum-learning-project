import os
import argparse

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

os.environ["WANDB_PROJECT"] = "grpo"
os.environ["WANDB_LOG_MODEL"] = "llama"

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from functools import partial, update_wrapper

from rewards import k_likelihood_reward_func, correctness_reward_func, vr_cli_reward_func, vr_cli_reward_func_hybrid, vr_cli_reward_func_absolute
from models import context_model, context_tokenizer
from data import context_dataset as dataset, context_eval_dataset as eval_dataset, context_train_dataset as train_dataset
from custom_grpo_trainer import CustomGRPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training script.")
    parser.add_argument('--reward_type', type=str, choices=['relative', 'absolute', 'hybrid'], default='absolute',
                        help='Type of reward to use (relative, absolute, or hybrid).')
    parser.add_argument('--dataset_name', type=str, default='gsm8k', help='Name of the dataset.')
    parser.add_argument('--model', type=str, default='base', help='Model to use.')
    return parser.parse_args()

args = parse_args()


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
    max_prompt_length = 512,
    max_completion_length = 1024,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps=300,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir=f"ppl_{args.reward_type}_{args.dataset_name}_{args.model}_full_test",
    eval_strategy = "steps",
    eval_steps = 100,
    eval_on_start = True,
    per_device_eval_batch_size = 1,
    save_strategy = "steps",
    save_steps = 500,
)

# hacky method, otherwise reward func metadata won't be preserved by partial
# rf = partial(vr_cli_reward_func, reward_type=args.reward_type, model=args.model)
# rf = update_wrapper(rf, vr_cli_reward_func)

rf = vr_cli_reward_func_hybrid if args.reward_type == 'hybrid' else (
    vr_cli_reward_func_absolute if args.reward_type == 'absolute' else vr_cli_reward_func
)

trainer = CustomGRPOTrainer(
    model=context_model,
    processing_class=context_tokenizer,
    reward_funcs=[rf],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_type=args.reward_type,
    model_type=args.model,
)

trainer.train()
trainer.evaluate()

context_model.save_lora(f"ppl_{args.reward_type}_{args.dataset_name}_{args.model}_{'full' if training_args.num_train_epochs else 'debug'}_test")
