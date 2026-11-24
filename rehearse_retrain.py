#!/usr/bin/env python3
"""
rehearse_retrain.py: find a test message that one retrain reliably fixes.

    python3 rehearse_retrain.py errors              # ModernBERT's mistakes on the test set, most confident first
    python3 rehearse_retrain.py try test-01234 ...  # retrain on each, with several seeds, and report
    python3 rehearse_retrain.py try test-01234 --params head --lr 1e-3 --steps 20   # another recipe

For each candidate it does what the live beat does (bert.retrain_one from the
original checkpoint, then the whole test set again) once per seed, and prints
whether the message flipped, how sure ModernBERT became, how long it took, and how many
other test messages went from right to wrong. Pick one that flips on every
seed and breaks nothing, and put it in demo.json under "retrain".
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import bert  # noqa: E402

SEEDS = (7, 8, 9)


def baseline(model, tok, test):
    path = os.path.join(common.MODELS, "bert", "test-predictions.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    preds = bert.predict(model, tok, [r["text"] for r in test])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    return preds


def cmd_errors(model, tok, test, _args):
    preds = baseline(model, tok, test)
    wrong = [(p, r) for p, r in zip(preds, test) if p["label"] != r["label"]]
    wrong.sort(key=lambda pr: -pr[0]["confidence"])
    print(f"{len(wrong)} wrong of {len(test)}")
    for p, r in wrong:
        print(f"{r['id']}  {p['confidence']:.2f}  said {p['label']:<34} gold {r['label']:<34} {r['text'][:90]}")


def cmd_try(model, tok, test, args):
    opts = {"steps": None, "lr": None, "params": None, "optimizer": None}
    ids = []
    it = iter(args)
    for a in it:
        if a in ("--steps", "--lr", "--params", "--optimizer"):
            v = next(it)
            opts[a[2:]] = int(v) if a == "--steps" else float(v) if a == "--lr" else v
        else:
            ids.append(a)
    print(f"recipe: {dict(bert.RETRAIN, **{k: v for k, v in opts.items() if v is not None})}")
    preds = baseline(model, tok, test)
    gold = [r["label"] for r in test]
    was = [p["label"] == g for p, g in zip(preds, gold)]
    train = common.split("train")
    by_id = {r["id"]: (i, r) for i, r in enumerate(test)}
    for tid in ids:
        i, row = by_id[tid]
        print(f"\n{tid}  gold {row['label']}  ModernBERT said {preds[i]['label']} ({preds[i]['confidence']:.2f})  {row['text']}")
        for seed in SEEDS:
            m = copy.deepcopy(model)
            t0 = time.time()
            bert.retrain_one(m, tok, row["text"], row["label"], train, seed=seed,
                             steps=opts["steps"], lr=opts["lr"], params=opts["params"],
                             optimizer=opts["optimizer"])
            t_train = time.time() - t0
            after_one = bert.predict(m, tok, [row["text"]], batch_size=1)[0]
            t1 = time.time()
            after = bert.predict(m, tok, [r["text"] for r in test])
            t_rerun = time.time() - t1
            now = [p["label"] == g for p, g in zip(after, gold)]
            broken = [test[j]["id"] for j in range(len(test)) if was[j] and not now[j]]
            fixed = [test[j]["id"] for j in range(len(test)) if now[j] and not was[j]]
            print(f"  seed {seed}: {'FLIPPED' if after_one['label'] == row['label'] else 'no flip':8} -> "
                  f"{after_one['label']} ({after_one['confidence']:.2f})  right {sum(was)} -> {sum(now)}  "
                  f"fixed {len(fixed)} broken {len(broken)} {broken[:5]}  train {t_train:.1f}s rerun {t_rerun:.1f}s",
                  flush=True)
            del m


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("errors", "try"):
        sys.exit(__doc__)
    model, tok = bert.load()
    test = common.split("test")
    {"errors": cmd_errors, "try": cmd_try}[cmd](model, tok, test, sys.argv[2:])


if __name__ == "__main__":
    main()
