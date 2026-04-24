#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """あなたは日本語の算数文章題を解くアシスタントです。
問題を丁寧に解き、最後の行では必ず指定された形式で最終解答だけを出力してください。"""


def normalize_answer(text: str) -> str | None:
    if text is None:
        return None
    text = text.strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = match.group(0)
    if "." in value:
        try:
            number = float(value)
            if number.is_integer():
                return str(int(number))
        except ValueError:
            pass
    return value


def parse_prediction(text: str, answer_prefix: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix_pattern = re.compile(rf"{re.escape(answer_prefix)}\s*[:：]\s*(.*)")

    for line in reversed(lines):
        match = prefix_pattern.search(line)
        if match:
            return normalize_answer(match.group(1))

    for line in reversed(lines):
        normalized = normalize_answer(line)
        if normalized is not None:
            return normalized
    return None


def build_prompt(example: dict, tokenizer) -> str:
    user_prompt = f'{example["instruction"]}\n\n{example["question"]}'
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--output-dir", default="results/mgsm_ja_vllm")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("CohereLabs/global-mgsm", "ja", split="test")
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
        max_tokens=512,
    )

    prompts = [build_prompt(example, tokenizer) for example in ds]
    answer_prefix = ds[0]["answer_prefix"]

    records = []
    correct = 0
    unparsable = 0

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_examples = ds.select(range(start, end))
        outputs = llm.generate(batch_prompts, sampling_params)

        for output, example in zip(outputs, batch_examples):
            text = output.outputs[0].text.strip()
            pred = parse_prediction(text, example["answer_prefix"])
            gold = normalize_answer(example["answer"])
            is_correct = pred == gold
            if pred is None:
                unparsable += 1
            if is_correct:
                correct += 1

            records.append(
                {
                    "question": example["question"],
                    "gold": gold,
                    "prediction": pred,
                    "raw_output": text,
                    "correct": is_correct,
                }
            )

    total = len(records)
    accuracy = correct / total if total else 0.0
    summary = {
        "model": args.model,
        "dataset": "CohereLabs/global-mgsm",
        "subset": "ja",
        "split": "test",
        "num_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "unparsable": unparsable,
        "answer_prefix": answer_prefix,
        "prompting": "zero-shot, non-thinking mode, final line formatted as 答え: <integer>",
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
