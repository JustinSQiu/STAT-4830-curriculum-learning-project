from trl import GRPOTrainer
from models import base_model, base_tokenizer
from rewards import compute_vr_cli_reward

class CustomGRPOTrainer(GRPOTrainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        # Use provided eval_dataset or default to self.eval_dataset
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
        import torch, math
        # Loop over evaluation samples
        for sample in eval_dataset:
            question = sample["question"]  # adapt if your dataset uses a different key for question
            true_answer = sample.get("answer") or sample.get("ground_truth")  # adapt key for ground truth
            # 1. Generate reasoning context with the policy model
            with torch.no_grad():
                # Standard generation using model.generate
                inputs = self.processing_class(question, return_tensors='pt').to(self.model.device)
                gen_out = self.model.generate(**inputs, max_new_tokens=128)  # limit context length
                context_text = self.processing_class.decode(gen_out[0], skip_special_tokens=True)
            # 2. Form combined prompt for the base model
            combined_prompt = f"{question}\n{context_text}\n"
            # 3. Generate final answer with base model (frozen)
            with torch.no_grad():
                base_inputs = base_tokenizer(combined_prompt, return_tensors='pt').to(base_model.device)
                ans_out = base_model.generate(**base_inputs, max_new_tokens=32)
                answer_text = base_tokenizer.decode(ans_out[0], skip_special_tokens=True).strip()
            # 4. Compare answer with ground truth for accuracy
            if true_answer is not None:
                # Simple normalization for comparison
                pred_norm = answer_text.strip().strip('.').lower()
                true_norm = true_answer.strip().strip('.').lower()
                if pred_norm == true_norm:
                    correct += 1
            # 5. Compute reward using custom reward function, if available
            reward_value = None
            reward_list = compute_vr_cli_reward(question=question, answer=true_answer, context=context_text)
            total_reward += reward_list[0] if isinstance(reward_list, (list, tuple)) else float(reward_list)
            # 6. Compute perplexity contributions: use base model to get NLL of its own answer
            if answer_text:
                with torch.no_grad():
                    full_input = base_tokenizer(combined_prompt + answer_text, return_tensors='pt')
                    input_ids = full_input.input_ids.to(base_model.device)
                    # Mask out everything except the answer portion for loss
                    prompt_len = len(base_tokenizer(combined_prompt)["input_ids"])
                    labels = input_ids.clone()
                    labels[:, :prompt_len] = -100
                    out = base_model(input_ids, labels=labels)
                    # out.loss is mean NLL over answer tokens
                    n_tokens = (labels != -100).sum().item()
                    if n_tokens > 0:
                        total_nll += out.loss.item() * n_tokens
                        total_tokens += n_tokens
            total += 1
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_reward = total_reward / total if total > 0 else 0.0
        avg_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        ppl = math.exp(avg_nll) if total_tokens > 0 else float('inf')
        metrics = {
            "eval_accuracy": accuracy,
            "eval_avg_reward": avg_reward,
            "eval_perplexity": ppl
        }
        self.log(metrics)
        return metrics
