#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """あなたは日本語の常識問題を解くアシスタントです。
与えられた5つの選択肢から最も適切なものを1つ選んでください。
回答は必ず 0, 1, 2, 3, 4 のいずれか1文字だけを出力してください。"""


def build_prompt(example: dict, tokenizer) -> str:
    user_prompt = f"""次の問題に答えてください。

問題: {example["question"]}
0. {example["choice0"]}
1. {example["choice1"]}
2. {example["choice2"]}
3. {example["choice3"]}
4. {example["choice4"]}

回答は数字1文字のみで出力してください。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_prediction(text: str) -> int | None:
    match = re.search(r"[0-4]", text)
    if match:
        return int(match.group(0))

    upper = text.upper()
    for letter, value in zip("ABCDE", range(5)):
        if letter in upper:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--output-dir", default="results/jcommonsenseqa_vllm")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("sbintuitions/JCommonsenseQA", split="validation")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=2048,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
        language_model_only=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=8,
        stop=["\n"],
    )

    prompts = [build_prompt(example, tokenizer) for example in ds]

    records = []
    correct = 0
    unparsable = 0

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating"):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_examples = ds.select(range(start, min(start + args.batch_size, len(ds))))
        outputs = llm.generate(batch_prompts, sampling_params)

        for output, example in zip(outputs, batch_examples):
            text = output.outputs[0].text.strip()
            pred = parse_prediction(text)
            gold = int(example["label"])
            is_correct = pred == gold
            if pred is None:
                unparsable += 1
            if is_correct:
                correct += 1

            record = {
                "q_id": example["q_id"],
                "question": example["question"],
                "gold": gold,
                "prediction": pred,
                "raw_output": text,
                "correct": is_correct,
            }
            records.append(record)

    total = len(records)
    accuracy = correct / total if total else 0.0
    summary = {
        "model": args.model,
        "dataset": "sbintuitions/JCommonsenseQA",
        "split": "validation",
        "num_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "unparsable": unparsable,
        "prompting": "zero-shot, non-thinking mode, answer with a single digit 0-4",
        "engine": "vLLM",
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
