from datasets import load_dataset, Dataset

SYSTEM_PROMPT_CONTEXT = """
Given the math question below, write a step-by-step reasoning context that clearly explains concepts needed to solve the problem.
"""

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions_for_context(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT_CONTEXT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data

context_dataset = get_gsm8k_questions_for_context()
context_train_dataset = get_gsm8k_questions_for_context(split="train")
context_eval_dataset = get_gsm8k_questions_for_context(split="test").select(range(100)) # type: ignore


def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
train_dataset = get_gsm8k_questions(split="train")
eval_dataset = get_gsm8k_questions(split="test").select(range(100)) # type: ignore
