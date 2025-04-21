import torch
import torch.nn.functional as F
import math

from models import base_model, base_tokenizer, context_model, context_tokenizer

def compute_stable_probability_reward(question, context, answer):
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")
    prompt = f"{question}\n{context}\nThe answer is: "
    full_input = f"{prompt}{answer}"

    # Tokenize inputs
    prompt_ids = base_tokenizer(prompt, return_tensors="pt").input_ids
    full_input_ids = base_tokenizer(full_input, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[-1]
    input_ids = full_input_ids.to(base_model.device)

    # Get logits
    with torch.no_grad():
        outputs = base_model(input_ids, output_hidden_states=True)
        
    # If outputs.logits is not the full logits, compute them using lm_head
    # I don't know why this happens but it does
    if outputs.logits.shape[-1] != base_model.config.vocab_size:
        # Apply the final projection (lm_head) to the last hidden state
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        logits = base_model.lm_head(hidden_states)   # [1, seq_len, vocab_size]
    else:
        logits = outputs.logits # [1, seq_len, vocab_size]

    vocab_size = base_model.config.vocab_size
    # print(f"Logits shape: {logits.shape}")
    # print(f"Vocab size: {vocab_size}")

    # Shift logits to align each logit[i] predicting token[i+1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    # The tokens we are predicting (shifted by 1)
    target_token_ids = input_ids[:, 1:]
    target_token_ids = torch.clamp(target_token_ids, min=0, max=vocab_size - 1)

    # Extract the answer tokens only
    answer_log_probs = log_probs[0, prompt_len - 1:, :].gather(
        1, target_token_ids[0, prompt_len - 1:].unsqueeze(-1)
    ).squeeze(-1)
    avg_log_prob = answer_log_probs.mean().item()

    print(f"Answer log probs: {answer_log_probs}")
    print(f"Avg log prob: {avg_log_prob}", flush=True)

    return avg_log_prob

def k_likelihood_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for prompt, comp, ans in zip(prompts, completions, answer):
        question = prompt[1]['content']
        context = comp[0]['content']
        reward = compute_stable_probability_reward(question, context, ans)
        rewards.append(reward)
    return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for prompt, comp, ans in zip(prompts, completions, answer):
        question = prompt[1]['content']
        context = comp[0]['content']
        generation_prompt = f"{question}\n{context}\nThe answer is: "
        inputs = base_tokenizer(generation_prompt, return_tensors="pt").input_ids.to(base_model.device)
        output_ids = base_model.generate(
            inputs, 
            max_new_tokens=50,  # adjust this value as needed based on expected answer length
            do_sample=False
        )
        generated_text = base_tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
        
        generated_answer = generated_text.strip().split("\n")[0].strip()
        provided_answer = ans.strip()
        if generated_answer.lower() == provided_answer.lower():
            reward = 1.0
        else:
            reward = 0.0
        
        rewards.append(reward)
    return rewards


def compute_ppl(prompt: str, continuation: str) -> float:
    full_input = prompt + continuation
    # Tokenize the prompt and full input separately
    prompt_ids = base_tokenizer(prompt, return_tensors="pt").input_ids
    full_input_ids = base_tokenizer(full_input, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[-1]
    input_ids = full_input_ids.to(base_model.device)

    with torch.no_grad():
        outputs = base_model(input_ids, output_hidden_states=True)
    # Check if the logits need to be projected through lm_head
    if outputs.logits.shape[-1] != base_model.config.vocab_size:
        hidden_states = outputs.hidden_states[-1]
        logits = base_model.lm_head(hidden_states)
    else:
        logits = outputs.logits

    vocab_size = base_model.config.vocab_size
    # Align logits: each logits[i] predicts token[i+1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_token_ids = input_ids[:, 1:]
    target_token_ids = torch.clamp(target_token_ids, min=0, max=vocab_size - 1)
    
    # Only take log probabilities corresponding to the continuation tokens.
    # Because of the shift, we index from (prompt_len - 1) onward.
    continuation_log_probs = log_probs[0, prompt_len - 1:, :].gather(
        1, target_token_ids[0, prompt_len - 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    avg_log_prob = continuation_log_probs.mean().item()
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def compute_vr_cli_reward(question: str, answer: str, context: str) -> float:
    """
    I = [1 - (PPL(y|x, a) / PPL(y|x))] * 100,
    
    I =
    0 if I < 0.05,
    0.5 if 0.05 ≤ I < 1,
    0.9 if 1 ≤ I < 2,
    1 if I ≥ 2.
    Here:
      - story_info is the input prompt x,
      - chapter is the gold continuation y,
      - detailed_plan is the generated reasoning a.
    """
    # Baseline: probability of the chapter given the story info only.
    baseline_prompt = f"{question}\n"
    # With reasoning: include the detailed plan. The wording 'The answer is:' is optional;
    # we include it here to mimic the style of your existing rewards.
    improved_prompt = f"{question}\n{context}\n"

    baseline_ppl = compute_ppl(baseline_prompt, answer)
    improved_ppl = compute_ppl(improved_prompt, answer)

    # Compute the improvement percentage I.
    # A positive I means that conditioning on the detailed plan lowered perplexity.
    I = (1 - improved_ppl / baseline_ppl) * 100

    # Apply thresholding as described in Equation (7)
    if I < 0.05:
        reward = 0.0
    elif I < 1:
        reward = 0.5
    elif I < 2:
        reward = 0.9
    else:
        reward = 1.0

    print("---- VR-CLI Reward Debug ----")
    print(f"Baseline PPL (y|x): {baseline_ppl}")
    print(f"Improved PPL (y|x, a): {improved_ppl}")
    print(f"Improvement I (%): {I}")
    print(f"Assigned Reward: {reward}", flush=True)
    print("-----------------------------")

    return reward

def vr_cli_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for prompt, comp, ans in zip(prompts, completions, answer):
        story_info = prompt[1]['content']
        chapter = comp[0]['content']
        reward = compute_vr_cli_reward(story_info, chapter, ans)
        rewards.append(reward)
    return rewards



# Reward functions
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func_orig(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
