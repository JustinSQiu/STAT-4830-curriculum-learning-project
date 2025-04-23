from trl import GRPOTrainer
from models import base_model, base_tokenizer
from rewards import compute_vr_cli_reward

import torch
import math

class CustomGRPOTrainer(GRPOTrainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        print("Evaluating...")
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval_dataset provided for evaluation.")

        # Set models to eval mode
        self.model.eval()
        base_model.eval()

        correct = 0
        total = 0
        total_reward = 0.0
        total_nll = 0.0
        total_tokens = 0

        for sample in eval_dataset:
            # Use full prompt template rather than raw question
            prompt = sample.get("prompt")
            true_answer = sample.get("answer") or sample.get("ground_truth")

            # 1. Generate reasoning context
            with torch.no_grad():
                inputs = self.processing_class(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_prompt_length,
                    padding='longest'
                ).to(self.model.device)

                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length
                )
                context_text = self.processing_class.decode(
                    gen_out[0], skip_special_tokens=True
                )

            # 2. Build combined text for base model
            # flatten prompt messages
            if isinstance(prompt, list):
                combined_prompt = "".join([m['content'] + "\n" for m in prompt])
            else:
                combined_prompt = prompt + "\n"
            combined_prompt += context_text + "\n"

            # 3. Generate final answer
            with torch.no_grad():
                base_inputs = base_tokenizer(
                    combined_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=base_model.config.max_position_embeddings,
                    padding='longest'
                ).to(base_model.device)

                ans_out = base_model.generate(
                    **base_inputs,
                    max_new_tokens=self.max_completion_length
                )
                answer_text = base_tokenizer.decode(
                    ans_out[0], skip_special_tokens=True
                ).strip()

            # 4. Accuracy
            if true_answer is not None:
                pred_norm = answer_text.strip().strip('.').lower()
                true_norm = true_answer.strip().strip('.').lower()
                if pred_norm == true_norm:
                    correct += 1

            # 5. Reward
            reward_list = compute_vr_cli_reward(
                question="".join([m['content'] for m in prompt]) if isinstance(prompt, list) else prompt,
                answer=true_answer,
                context=context_text
            )
            total_reward += (reward_list[0] if isinstance(reward_list, (list, tuple)) else float(reward_list))

            # 6. Perplexity / NLL on the base model's own answer
            if answer_text:
                full_input = base_tokenizer(
                    combined_prompt + answer_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=base_model.config.max_position_embeddings,
                    padding='longest'
                )
                input_ids = full_input.input_ids.to(base_model.device)

                # recompute prompt length on truncated prompt only
                prompt_tok = base_tokenizer(
                    combined_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=base_model.config.max_position_embeddings
                )
                prompt_len = prompt_tok.input_ids.shape[-1]

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                out = base_model(input_ids, labels=labels)
                n_tokens = (labels != -100).sum().item()
                if n_tokens > 0:
                    total_nll += out.loss.item() * n_tokens
                    total_tokens += n_tokens

            total += 1

        # aggregate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_reward = total_reward / total if total > 0 else 0.0
        avg_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        ppl = math.exp(avg_nll) if total_tokens > 0 else float('inf')

        metrics = {
            "eval_accuracy": accuracy,
            "eval_avg_reward": avg_reward,
            "eval_perplexity": ppl,
        }
        self.log(metrics)
        return metrics
