# Data

Banking77 by PolyAI (Casanueva et al., 2020), licensed CC BY 4.0:
<https://huggingface.co/datasets/PolyAI/banking77>, at revision
`1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`. Paper: "Efficient Intent Detection
with Dual Sentence Encoders", <https://arxiv.org/abs/2003.04807>.

| File | |
|---|---|
| `train.jsonl`, `test.jsonl` | The dataset's own split, 10,003 and 3,080 messages, one `{"id", "text", "label"}` per line, written by `../prepare_data.py` from the parquet files at that revision. Text is the dataset's, trimmed of surrounding whitespace |
| `labels.json` | The 77 intent names in the dataset's index order |
| `intents.json` | Ours: a one-line description per intent, which columns A and C see |
