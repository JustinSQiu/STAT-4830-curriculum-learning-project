from trl import GRPOTrainer
from models import base_model, base_tokenizer, small_base_model, small_base_tokenizer
from rewards import compute_vr_cli_reward
import torch.nn.functional as F
from vllm import SamplingParams
from trl.data_utils import apply_chat_template

from data import SYSTEM_PROMPT_CONTEXT
import torch
import math

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.reward_type = kwargs.get("reward_type", "relative")
        del kwargs["reward_type"]
        self.model_type = kwargs.get("model_type", "base")
        del kwargs["model_type"]
        if self.reward_type not in ["relative", "absolute", "hybrid"]:
            raise ValueError("Invalid reward type. Choose from 'relative', 'absolute', or 'hybrid'.")
        super().__init__(*args, **kwargs)
        # Prepare vLLM sampling parameters
        self._ctx_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.max_completion_length,
        )
        self._ans_sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.max_completion_length,
        )
        self.base_model = base_model if self.model_type == "base" else small_base_model
        self.base_tokenizer = base_tokenizer if self.model_type == "base" else small_base_tokenizer

    def evaluate(self, eval_dataset=None, **kwargs):
        print("Evaluating...")
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval_dataset provided for evaluation.")

        # Set models to eval mode
        self.model.eval()
        self.base_model.eval()

        correct = 0
        total = 0
        total_reward = 0.0
        # total_nll = 0.0
        # total_tokens = 0

        for sample in eval_dataset:
            prompt = sample.get("question")
            true_answer = sample.get("answer")

            # 1. Generate reasoning context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_CONTEXT},
                {"role": "user",   "content": prompt},
            ]
            t = apply_chat_template({"messages": messages}, self.processing_class)
            context_text = self.model.fast_generate(
                t["text"],
                sampling_params=self._ctx_sampling_params,
            )[0].outputs[0].text.strip()

            messages = [
                {"role": "system", "content": "Generate only the answer. Do not include any other text besides the final number."},
                {"role": "user", "content": f"{prompt}\n{context_text}\nThe final answer is: "},
                # {"role": "user", "content": f"{prompt}\n{context_text}"},
            ]
            t = apply_chat_template({"messages": messages}, self.processing_class)
            answer_text = self.base_model.fast_generate(
                t["text"],
                sampling_params=self._ans_sampling_params,
            )[0].outputs[0].text.strip()


            ans_prompt = f"{prompt}\n{context_text}\n"
            answer_text = self.base_model.fast_generate(
                ans_prompt,
                sampling_params=self._ans_sampling_params,
            )[0].outputs[0].text.strip()

            print(f"Predicted Answer: {answer_text}")

            # with torch.no_grad():
            #     inputs = self.processing_class( # encode
            #         prompt,
            #         return_tensors='pt',
            #         truncation=True,
            #         max_length=self.max_prompt_length,
            #         padding='longest'
            #     ).to(self.model.device)

            #     gen_out = self.model.generate( # generate
            #         **inputs,
            #         max_new_tokens=self.max_completion_length
            #     )
            #     context_text = self.processing_class.decode( # decode
            #         gen_out[0], skip_special_tokens=True
            #     )

            # # 2. Build combined text for base model
            # # flatten prompt messages
            # if isinstance(prompt, list):
            #     combined_prompt = "".join([m['content'] + "\n" for m in prompt])
            # else:
            #     combined_prompt = prompt + "\n"
            # combined_prompt += context_text + "\n"

            # # 3. Generate final answer
            # with torch.no_grad():
            #     base_inputs = base_tokenizer(
            #         combined_prompt,
            #         return_tensors='pt',
            #         truncation=True,
            #         max_length=base_model.config.max_position_embeddings,
            #         padding='longest'
            #     ).to(base_model.device)

            #     ans_out = base_model.generate(
            #         **base_inputs,
            #         max_new_tokens=self.max_completion_length
            #     )
            #     answer_text = base_tokenizer.decode(
            #         ans_out[0], skip_special_tokens=True
            #     ).strip()

            # 4. Accuracy
            if true_answer:
                pred_norm = answer_text.strip().strip('.').lower()
                true_norm = true_answer.strip().strip('.').lower()
                if pred_norm == true_norm:
                    correct += 1

            # 5. Reward
            reward = compute_vr_cli_reward(
                question=prompt,
                answer=true_answer,
                context=context_text,
                tpe=self.reward_type,
            )
            total_reward += reward

            # # 6. Compute NLL manually instead of using fused cross-entropy
            # if answer_text:
            #     # re-tokenize and truncate prompt+answer to model max
            #     # 1. Tokenize combined prompt + answer just once:
            #     full = base_tokenizer(
            #         combined_prompt + answer_text,
            #         return_tensors="pt",
            #         truncation=True,
            #         max_length=base_model.config.max_position_embeddings,
            #     )
            #     input_ids = full.input_ids.to(base_model.device)  # (1, L)
            #     # 2. Decide where the prompt ends by re-tokenizing only the prompt *with the same settings*:
            #     prompt_only = base_tokenizer(
            #         combined_prompt,
            #         return_tensors="pt",
            #         truncation=True,
            #         max_length=base_model.config.max_position_embeddings,
            #     )
            #     prompt_len = prompt_only.input_ids.shape[-1]
            #     # 3. Run the model once on full `input_ids`:
            #     with torch.no_grad():
            #         outputs = base_model(input_ids, output_hidden_states=True)
            #     logits = outputs.logits  # (1, L, V)
            #     log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            #     # 4. Slice answer tokens consistently:
            #     answer_log_probs = log_probs[0, prompt_len - 1 : , :].gather(
            #         1,
            #         input_ids[0, prompt_len : ].unsqueeze(-1)  # same `input_ids` as logits
            #     ).squeeze(-1)


            #     # accumulate NLL and token count
            #     total_nll    += -answer_log_probs.sum().item()
            #     total_tokens += answer_log_probs.numel()

            total += 1
        # aggregate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_reward = total_reward / total if total > 0 else 0.0
        # avg_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        # ppl = math.exp(avg_nll) if total_tokens > 0 else float('inf')

        metrics = {
            "eval_accuracy": accuracy,
            "eval_avg_reward": avg_reward,
            # "eval_perplexity": ppl,
        }
        self.log(metrics)
        return metrics
