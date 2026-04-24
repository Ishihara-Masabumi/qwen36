# vLLMによるJapanese GSM8K (MGSM日本語版) 評価結果

## 目的

Qwen3.6-35B-A3B 系モデルを、日本語算数推論ベンチマーク Japanese GSM8K (MGSM日本語版) で再現可能な形で評価する。

## 内容

- データセット: `CohereLabs/global-mgsm` subset `ja` test split
- 推論エンジン: `vllm`
- モデル: `Qwen/Qwen3.6-35B-A3B-FP8`
- 実行環境: `conda` 環境 `qwen36`
- 評価件数: `250`
- 推論設定: `language_model_only=True`, `enable_thinking=False`
- 生成設定: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `max_tokens=512`
- 出力形式: 最終行を `答え: <integer>` に統一
- 実行スクリプト: `scripts/eval_mgsm_ja_vllm.py`

## 結果

- 正解数: `215 / 250`
- Accuracy: `0.8600`
- Unparsable: `0`

## 結果評価

- zero-shot 条件で `86.0%` の accuracy となり、日本語算数推論でも高い水準の性能を示した
- 失敗例は、割引条件の解釈、損益分岐の年数、逆算を要する文章題など、多段の数量関係を正しく式に落とす必要がある問題に集中していた
- 初期設定では生成長が不足して最終行まで到達しないケースがあったため、`max_tokens=512` に拡張して評価を安定化させた
