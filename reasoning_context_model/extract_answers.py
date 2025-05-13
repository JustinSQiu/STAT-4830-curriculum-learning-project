import re
import csv
import json
import argparse
from helpers import get_gpt_response

def parse_entries(text):
    raw_entries = text.split('---- VR-CLI Reward Debug ----')
    entries = []
    for raw in raw_entries:
        block = raw.strip()
        if not block:
            continue

        # extract question
        q_match = re.search(r'Question:\s*(.*?)\s*(?:Baseline Prompt:|$)', block, re.DOTALL)
        if not q_match:
            continue
        question = q_match.group(1).strip().replace('\n', ' ')

        # first (student) answer
        answers = re.findall(r'^Answer:\s*(.+)$', block, re.MULTILINE)
        if not answers:
            continue
        first_answer = answers[0].strip()

        # multi‐line predicted (true) answer
        pred_match = re.search(
            r'Predicted Answer:\s*(.*?)\s*(?=Question:|$)',
            block,
            re.DOTALL
        )
        if not pred_match:
            continue
        true_answer = pred_match.group(1).strip()

        entries.append((question, first_answer, true_answer))
    return entries

def evaluate_responses(rows):
    """
    For each (question, student, true) tuple, ask GPT if it's correct.
    Returns a list of booleans, same length as rows.
    """
    correctness = []
    for question, student_ans, true_ans in rows:
        grader_prompt = [
            {'role': 'system', 'content': (
                "You are an assistant that checks whether a student's answer correctly matches "
                "the true answer for a math word problem. "
                "User will send a JSON with 'question', 'student_answer', and 'true_answer'. "
                "You must return JSON exactly of the form: {\"correct\": true} "
                "or {\"correct\": false}."
            )},
            {'role': 'user', 'content': json.dumps({
                'question': question,
                'student_answer': student_ans,
                'true_answer': true_ans
            }, ensure_ascii=False)}
        ]

        raw = get_gpt_response(grader_prompt).strip()
        raw = re.sub(r"^```(\w+)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            result = json.loads(raw)
            is_correct = bool(result.get('correct'))
        except (json.JSONDecodeError, TypeError):
            # if GPT responds oddly, count as false
            is_correct = False

        correctness.append(is_correct)
    return correctness

def write_csv(rows, correctness_flags, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'true_answer', 'correctness'])
        for (q, a, t), flag in zip(rows, correctness_flags):
            writer.writerow([q, a, t, str(flag)])

def main():
    parser = argparse.ArgumentParser(
        description="1) Extract Q&A blocks into CSV  2) Use GPT to mark each as correct/incorrect"
    )
    parser.add_argument('input_file',  help='Path to the input log text file')
    parser.add_argument('output_csv',  help='Path for the extracted CSV')
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    rows = parse_entries(raw_text)
    if not rows:
        print("No entries found. Make sure your file uses the VR-CLI separator and has at least one 'Answer:' and a 'Predicted Answer:' per block.")
        return

    # evaluate each row
    correctness_flags = evaluate_responses(rows)

    # write out CSV with an extra column
    write_csv(rows, correctness_flags, args.output_csv)
    print(f"Wrote {len(rows)} rows (with correctness) to {args.output_csv}")

    correct_count = sum(correctness_flags)
    total = len(rows)
    print(f"{correct_count}/{total} responses are correct.")

if __name__ == '__main__':
    main()
