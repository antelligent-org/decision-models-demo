#!/usr/bin/env python3
"""
prepare_data.py: turn the Banking77 parquet files into the JSONL this demo reads.

    python3 prepare_data.py DIR      # DIR holds train-*.parquet, test-*.parquet, README.md

Download them first, pinned to the revision in common.DATASET:

    https://huggingface.co/datasets/PolyAI/banking77/resolve/<revision>/data/train-00000-of-00001.parquet

The output (data/train.jsonl, data/test.jsonl, data/labels.json) is checked in,
so nothing here needs the network on the day. Needs pyarrow.
"""

import json
import os
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA  # noqa: E402


def label_names(readme):
    """The class names, in index order, from the dataset card's YAML header."""
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    found = re.findall(r"^\s+'(\d+)': (.+)$", text, flags=re.M)
    names = {int(i): n.strip() for i, n in found}
    if sorted(names) != list(range(77)):
        sys.exit(f"expected 77 labels in {readme}, found {len(names)}")
    return [names[i] for i in range(77)]


def main():
    src = sys.argv[1]
    names = label_names(os.path.join(src, "README.md"))
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump(names, fh, indent=1)
        fh.write("\n")
    for split in ("train", "test"):
        table = pq.read_table(os.path.join(src, f"{split}-00000-of-00001.parquet")).to_pylist()
        path = os.path.join(DATA, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for i, row in enumerate(table):
                fh.write(json.dumps({"id": f"{split}-{i:05d}", "text": row["text"].strip(),
                                     "label": names[row["label"]]}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(table)} rows -> {path}")


if __name__ == "__main__":
    main()
