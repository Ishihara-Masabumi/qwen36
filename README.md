# Qwen 3.6 セットアップ

このディレクトリには、Qwen 3.6 の推論環境と JCommonsenseQA 評価を再現するための最小構成ファイルが入っています。

## 想定構成

- `/home/ubuntu/qwen36/scripts/eval_jcommonsenseqa_vllm.py`
- `/home/ubuntu/qwen36/scripts/eval_jcommonsenseqa_vllm.sh`
- `/home/ubuntu/qwen36/reports/jcommonsenseqa_vllm_report.md`
- `/home/ubuntu/qwen36/results/jcommonsenseqa_vllm/summary.json`
- `conda` environment: `qwen36`

## デフォルトモデル

- `Qwen/Qwen3.6-35B-A3B-FP8`

## すぐに試す

```bash
conda activate qwen36
cd /home/ubuntu/qwen36
python scripts/eval_jcommonsenseqa_vllm.py --limit 50 --output-dir results/jcommonsenseqa_vllm_smoke
```

## 本評価を回す

```bash
conda activate qwen36
cd /home/ubuntu/qwen36
./scripts/eval_jcommonsenseqa_vllm.sh
```

## 現在の評価結果

- ベンチマーク: `sbintuitions/JCommonsenseQA` validation
- モデル: `Qwen/Qwen3.6-35B-A3B-FP8`
- 推論エンジン: `vLLM`
- 正解数: `1067 / 1119`
- Accuracy: `0.9535`
- Unparsable: `0`

## 参考レポート

- `reports/jcommonsenseqa_vllm_report.md`
- `reports/issue_1_vllm_jcommonsenseqa.md`
