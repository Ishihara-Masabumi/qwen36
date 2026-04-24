# Japanese GSM8K (MGSM日本語版) 評価結果

## 目的

Qwen/Qwen3.6-35B-A3B-FP8 を vLLM で実行し、日本語算数推論ベンチマーク Japanese GSM8K (MGSM日本語版) における基礎性能を確認する。

## 内容

- データセット: `CohereLabs/global-mgsm` subset `ja` test split
- 件数: `250`
- モデル: `Qwen/Qwen3.6-35B-A3B-FP8`
- 推論エンジン: `vLLM 0.19.1`
- 実行設定: `language_model_only=True`, `enable_thinking=False`
- 生成設定: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `max_tokens=512`
- 出力形式: 最終行を `答え: <integer>` に統一
- 実行スクリプト: `scripts/eval_mgsm_ja_vllm.py`
- 出力ファイル:
  - `results/mgsm_ja_vllm/summary.json`
  - `results/mgsm_ja_vllm/predictions.jsonl`

## 結果

- 正解数: `215 / 250`
- Accuracy: `0.8600`
- Unparsable: `0`

## 結果評価

- zero-shot で `86.0%` の accuracy を記録し、日本語算数文章題に対して高い推論性能を確認できた
- 失敗例は、複数条件の割引、逆算、年数条件のような多段の数量関係を含む問題に集中していた
- 初期設定では生成長不足によって最終回答行が切れるケースがあったため、`max_tokens=512` に調整して安定化した
