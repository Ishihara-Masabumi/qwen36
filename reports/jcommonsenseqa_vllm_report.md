# vLLMによるJCommonsenseQA評価結果

## 目的

Qwen3.6-35B-A3B 系モデルを、日本語常識推論ベンチマーク JCommonsenseQA で再現可能な形で評価する。

## 内容

- データセット: `sbintuitions/JCommonsenseQA` validation split (1,119件)
- 推論エンジン: `vllm`
- モデル: `Qwen/Qwen3.6-35B-A3B-FP8`
- 実行環境: `conda` 環境 `qwen36`
- 推論設定: `language_model_only=True`, `enable_thinking=False`, `temperature=0.7`, `top_p=0.8`, `top_k=20`
- プロンプト方式: zero-shot、回答を `0` から `4` の数字1文字に限定
- 実行スクリプト: `scripts/eval_jcommonsenseqa_vllm.py`
- 補足: `lm-eval` 標準タスク `ja_leaderboard_jcommonsenseqa` はこの環境では廃止済みデータセット ID (`Rakuten/JGLUE`) を参照して失敗したため、同じ vLLM 構成で JCommonsenseQA を直接評価した

## 結果

- 正解数: `1067 / 1119`
- Accuracy: `0.9535`
- Unparsable: `0`

## 結果評価

- zero-shot かつ非思考モードでも `95.35%` に到達しており、日本語常識推論タスクに対して非常に高い精度を示した
- 失敗例は語感・語彙知識・日本語固有の言い回しに寄った問題が中心で、単純な出力形式崩れは見られなかった
- 初回実行では思考モードが有効なままでスコアが大きく崩れたが、`enable_thinking=False` に切り替えることで安定した。Qwen3.6 系の評価では思考モード制御が重要
