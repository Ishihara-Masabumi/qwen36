# vLLMによるJCommonsenseQA評価結果

## 目的

Qwen/Qwen3.6-35B-A3B-FP8 を vLLM で実行し、日本語常識推論ベンチマーク JCommonsenseQA における基礎性能を確認する。

## 内容

- データセット: `sbintuitions/JCommonsenseQA` validation split
- 件数: 1,119
- モデル: `Qwen/Qwen3.6-35B-A3B-FP8`
- 推論エンジン: `vLLM 0.19.1`
- 実行設定: `language_model_only=True`, `enable_thinking=False`
- 生成設定: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `max_tokens=8`
- プロンプト方式: zero-shot、回答を `0` から `4` の数字1文字に限定
- 実行スクリプト: `scripts/eval_jcommonsenseqa_vllm.py`
- 出力ファイル:
  - `results/jcommonsenseqa_vllm/summary.json`
  - `results/jcommonsenseqa_vllm/predictions.jsonl`

## 結果

- 正解数: `1067 / 1119`
- Accuracy: `0.9535`
- Unparsable: `0`

## 結果評価

- zero-shot でも `95.35%` の accuracy となり、JCommonsenseQA に対してかなり高い性能を確認できた
- 失敗例は語彙知識や語感に依存する日本語固有の問題が中心で、出力フォーマット崩れは見られなかった
- 初回は思考モード有効のままで回答が崩れたため、Qwen3.6 系をベンチマークするときは `enable_thinking=False` の明示が重要
- `lm-eval` の既存タスク `ja_leaderboard_jcommonsenseqa` はこの環境では古いデータセット ID を参照して失敗したため、今回は vLLM を直接呼ぶスクリプトで評価した
