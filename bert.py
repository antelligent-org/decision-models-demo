#!/usr/bin/env python3
"""
bert.py: the ModernBERT model, ModernBERT-base fine-tuned as a 77-way classifier.

    python3 bert.py train                 # fine-tune on the 10,003 training messages
    python3 bert.py eval                  # batched accuracy on the 3,080 test messages
    python3 bert.py ask "my card hasn't arrived"

The checkpoint goes to models/bert/ (gitignored, about 600 MB). Training uses
the Apple GPU (MPS) when there is one. The recipe is fixed in RECIPE and was
not tuned on the test set: there is no early stopping and no model selection.

The same functions serve bench.py, serve.py and the retrain beat, which
continues training from the checkpoint for a few steps (train_steps).
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Keep the MPS allocator's cache on a short leash. By default it only gives
# memory back near the machine's limit, and a training run grew by about 1 GB
# every 100 steps until a 16 GB laptop swapped and stalled. These are fractions
# of the GPU's recommended working set (about 11 GB on a 16 GB M4): free cached
# blocks above 40%, and fail rather than swap above 75%.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

BASE = "answerdotai/ModernBERT-base"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
CHECKPOINT = os.path.join(common.MODELS, "bert")
RECIPE = {"epochs": 5, "batch_size": 32, "learning_rate": 5e-5, "weight_decay": 0.01,
          "warmup_ratio": 0.06, "max_length": 128, "seed": 13}


def encode(tok, texts, dev):
    """Tokenise, padding to a multiple of 16 so the GPU sees a few shapes rather than one per batch.

    On Apple silicon every new shape grows the MPS allocator's cache; with one
    shape per batch a training run reached 11 GB and pushed a 16 GB laptop into swap.
    """
    return tok(texts, padding=True, truncation=True, max_length=RECIPE["max_length"],
               pad_to_multiple_of=16, return_tensors="pt").to(dev)


def release(dev):
    """Hand cached GPU memory back to the system (the laptop has other work to do)."""
    if dev.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(path=CHECKPOINT, dev=None):
    """(model, tokenizer) from a checkpoint directory, in eval mode on dev."""
    dev = dev or device()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(dev).eval()
    return model, tok


def meta(path=CHECKPOINT):
    with open(os.path.join(path, "decision-models.json"), encoding="utf-8") as fh:
        return json.load(fh)


@torch.no_grad()
def predict(model, tok, texts, batch_size=64, top=3):
    """[{label, confidence, top: [[label, p], ...]}] for each text."""
    dev = next(model.parameters()).device
    id2label = model.config.id2label
    out = []
    for i in range(0, len(texts), batch_size):
        enc = encode(tok, texts[i:i + batch_size], dev)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
        values, idx = probs.topk(top, dim=-1)
        for v, ix in zip(values.tolist(), idx.tolist()):
            out.append({"label": id2label[ix[0]], "confidence": v[0],
                        "top": [[id2label[j], p] for j, p in zip(ix, v)]})
    if dev.type == "mps":
        torch.mps.synchronize()
    if len(texts) > batch_size:
        release(dev)
    return out


def ask_one(model, tok, text):
    """One message, timed as one call: tokenise, forward, softmax, back on the CPU."""
    start = time.perf_counter()
    result = predict(model, tok, [text], batch_size=1)[0]
    result["ms"] = (time.perf_counter() - start) * 1000
    return result


def train_steps(model, tok, examples, steps, lr, batch_size, seed, warmup=0, log=None):
    """Plain AdamW with linear warmup and decay over `steps` batches drawn from `examples`.

    examples is a list of (text, label_name). Batches are drawn by shuffling and
    cycling through the list, so a list with an example repeated in it sees that
    example in proportion. Returns the per-step losses.
    """
    dev = next(model.parameters()).device
    label2id = model.config.label2id
    rng = random.Random(seed)
    torch.manual_seed(seed)
    order = []
    no_decay = ("bias", "norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n.lower() for k in no_decay)],
         "weight_decay": RECIPE["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n.lower() for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)))
    model.train()
    losses, start = [], time.time()
    for step in range(steps):
        batch = []
        while len(batch) < batch_size:
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            batch.append(examples[order.pop()])
        enc = encode(tok, [t for t, _ in batch], dev)
        y = torch.tensor([label2id[lab] for _, lab in batch], device=dev)
        loss = model(**enc, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 25 == 0:
            release(dev)
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if log and (step + 1) % log == 0:
            recent = losses[-log:]
            gpu = f"  gpu {torch.mps.driver_allocated_memory() / 2**30:.1f} GB" if dev.type == "mps" else ""
            print(f"  step {step + 1:>5}/{steps}  loss {sum(recent) / len(recent):.4f}  "
                  f"{time.time() - start:6.0f}s{gpu}", flush=True)
    model.eval()
    release(dev)
    return losses


# Chosen in rehearsal (rehearse_retrain.py): on the demo's message this fixes it
# on every seed tried, and leaves the test-set total within one of where it was.
RETRAIN = {"steps": 10, "batch_size": 32, "learning_rate": 0.02, "seed": 7, "params": "all", "optimizer": "sgd"}
HEAD = ("head.", "classifier.")


def retrain_one(model, tok, text, label, train_rows, steps=None, lr=None, seed=None, on_step=None,
                params=None, optimizer=None):
    """Continue training on one corrected message, mixed into a fresh sample of the training data.

    Every batch holds the corrected message once plus batch_size - 1 training
    messages drawn at random, so the model learns the new example without being
    pushed to forget the old ones. Returns the per-step losses.
    """
    steps = steps or RETRAIN["steps"]
    lr = lr or RETRAIN["learning_rate"]
    params = params or RETRAIN["params"]
    optimizer = optimizer or RETRAIN["optimizer"]
    seed = RETRAIN["seed"] if seed is None else seed
    bs = RETRAIN["batch_size"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    label2id = model.config.label2id
    # "head" trains only the classifier on top of the encoder, so what the
    # encoder makes of every other message cannot move; "all" trains everything.
    trainable = [p for n, p in model.named_parameters() if params == "all" or n.startswith(HEAD)]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)
    # AdamW's first steps move every weight by about lr, whatever its gradient,
    # which disturbs answers far from the corrected message. Plain SGD moves each
    # weight in proportion to its gradient, so the change stays local.
    if optimizer == "sgd":
        opt = torch.optim.SGD(trainable, lr=lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=RECIPE["weight_decay"])
    model.train()
    losses = []
    for step in range(steps):
        batch = [(text, label)] + [(r["text"], r["label"]) for r in rng.sample(train_rows, bs - 1)]
        enc = encode(tok, [t for t, _ in batch], dev)
        y = torch.tensor([label2id[lab] for _, lab in batch], device=dev)
        loss = model(**enc, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if on_step:
            on_step(step + 1, steps, loss.item())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    del opt
    release(dev)
    return losses


def cmd_train(_args):
    names = common.labels()
    train = [(r["text"], r["label"]) for r in common.split("train")]
    dev = device()
    print(f"device {dev}, {len(train)} training messages, recipe {RECIPE}")
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=BASE_REVISION, num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)})
    model.to(dev)
    per_epoch = -(-len(train) // RECIPE["batch_size"])
    steps = per_epoch * RECIPE["epochs"]
    start = time.time()
    losses = train_steps(model, tok, train, steps, RECIPE["learning_rate"], RECIPE["batch_size"],
                         RECIPE["seed"], warmup=int(steps * RECIPE["warmup_ratio"]), log=50)
    seconds = time.time() - start
    os.makedirs(CHECKPOINT, exist_ok=True)
    stale = os.path.join(CHECKPOINT, "test-predictions.json")  # belongs to the previous checkpoint
    if os.path.exists(stale):
        os.remove(stale)
    model.save_pretrained(CHECKPOINT)
    tok.save_pretrained(CHECKPOINT)
    info = {"base": BASE, "base_revision": BASE_REVISION, "recipe": RECIPE, "steps": steps,
            "train_examples": len(train), "train_seconds": round(seconds, 1), "device": str(dev),
            "torch": torch.__version__, "final_loss": round(sum(losses[-50:]) / 50, 5),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(CHECKPOINT, "decision-models.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))


def cmd_eval(_args):
    model, tok = load()
    test = common.split("test")
    start = time.time()
    preds = predict(model, tok, [r["text"] for r in test])
    right = sum(p["label"] == r["label"] for p, r in zip(preds, test))
    sure = [(p, r) for p, r in zip(preds, test) if p["confidence"] > 0.9]
    print(f"accuracy {right}/{len(test)} = {right / len(test):.4f}  ({time.time() - start:.1f}s batched)")
    print(f"confidence > 0.9: {len(sure)}/{len(test)}, of which right "
          f"{sum(p['label'] == r['label'] for p, r in sure)}")


def cmd_ask(args):
    model, tok = load()
    ask_one(model, tok, "warm up")
    print(json.dumps(ask_one(model, tok, " ".join(args)), indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"train": cmd_train, "eval": cmd_eval, "ask": cmd_ask}.get(cmd, lambda _a: sys.exit(__doc__))(sys.argv[2:])
