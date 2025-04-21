from datasets import load_dataset, Dataset

# Preparing the dataset
SYSTEM_PROMPT = """
Given the math question below, write a step-by-step reasoning context that clearly explains concepts needed to solve the problem.
"""
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions_for_context(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data

dataset = get_gsm8k_questions_for_context()
eval_dataset = get_gsm8k_questions_for_context(split="test").select(range(50)) # type: ignore
