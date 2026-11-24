#!/usr/bin/env python3
"""
bench.py: run the three models over the 3,080 test messages, and write RESULTS.md.

    python3 bench.py run                             # all three, then the report
    python3 bench.py run --only modernbert           # one model (llm, modernbert, fern, or a list: llm,fern)
    python3 bench.py run --resume runs/2026-11-24T0700Z
    python3 bench.py report runs/2026-11-24T0700Z

Every message is sent one at a time, the way a support inbox would send them:
no batching, a kept-alive connection, and a few warm-up calls first so the
first timed call does not pay for loading or a TLS handshake. The LLM and Fern
run in parallel with each other (two different hosts, both waiting on the
network); ModernBERT runs on its own so nothing else competes for the machine
while it is timed.

Each model writes one JSONL line per message to runs/<timestamp>/, as it goes,
so a run that loses the network can be resumed. A call that still fails after
three tries is recorded as an error and scored as wrong.

The report also copies the LLM's and Fern's raw answers to cache/, which the
page replays, labelled "cached", when the network fails.
"""

import argparse
import json
import math
import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

THRESHOLD = 0.9
CACHE = os.path.join(common.HERE, "cache")
MODELS = ("llm", "modernbert", "fern")
TITLES = {"llm": "LLM, prompt only", "modernbert": "ModernBERT-base, fine-tuned", "fern": "Fern, fine-tuned"}
SHORT = {"llm": "LLM", "modernbert": "ModernBERT", "fern": "Fern"}

# ModernBERT runs on your own machine, so its compute seconds are priced at a cloud
# instance with the chip they were measured on. On demand, us-east-1, from
# instances.vantage.sh, 2026-11-23.
INSTANCE = {"instance": "AWS EC2 mac-m4.metal", "chip": "Apple M4 (10 vCPU, 24 GiB)", "usd_per_hour": 1.230,
            "source": "https://instances.vantage.sh/aws/ec2/mac-m4.metal",
            "note": "Same chip as the laptop ModernBERT was timed on. AWS bills Mac hosts per second after a "
                    "24-hour minimum, so this per-call figure assumes the machine is kept busy."}

# Fern is not billed by compute, tokens or calls: Fernfly charges a flat monthly fee.
# As published at the source on 2026-11-24.
PLAN = {"plan": "Fernfly Pro", "usd_per_month": 49, "terms": "unlimited requests, subject to fair use",
        "source": "https://fernfly.com/pricing", "as_of": "2026-11-24"}


def now_stamp():
    return time.strftime("%Y-%m-%dT%H%MZ", time.gmtime())


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {row["id"] for row in common.read_jsonl(path)}


def with_retries(fn, text, tries=3):
    last = None
    for attempt in range(tries):
        try:
            return fn(text)
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(2 * (attempt + 1))
    return {"error": str(last)[:500]}


def run_model(name, rows, path, ask, meta_path, meta, progress):
    have = done_ids(path)
    todo = [r for r in rows if r["id"] not in have]
    old = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    meta = dict(meta, started_utc=old.get("started_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(todo):
            out = with_retries(ask, r["text"])
            out.update(id=r["id"], gold=r["label"])
            out.setdefault("label", None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                progress(name, len(have) + i + 1, len(rows))
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def make_llm():
    import llm
    client = llm.client()
    for text in ("hello", "my card is lost", "how do I top up"):
        client.ask(text)
    meta = {"model": "llm", "config": client.cfg, "prompt_version": llm.PROMPT_VERSION,
            "prices": llm.PRICES[client.cfg["model"]], "prices_as_of": llm.PRICES_AS_OF,
            "prices_url": llm.PRICES_URL[client.cfg["provider"]],
            "system_prompt": llm.system_prompt(client.cfg["descriptions"])}
    return client.ask, meta


def make_fern(base):
    import fern
    client = fern.Client(base)
    for text in ("hello", "my card is lost", "how do I top up"):
        client.ask(text)
    return client.ask, {"model": "fern", "base": base, "entry": client.entry, "base_name": fern.BASES[base]["name"]}


def run_modernbert(rows, path, meta_path, progress):
    import bert
    model, tok = bert.load()
    for _ in range(10):
        bert.ask_one(model, tok, "warming up the model")
    meta = {"model": "modernbert", "checkpoint": bert.meta(), "device": str(bert.device())}

    def ask(text):
        out = bert.ask_one(model, tok, text)
        return {"label": out["label"], "confidence": out["confidence"], "ms": out["ms"], "top": out["top"]}

    run_model("modernbert", rows, path, ask, meta_path, meta, progress)


def run_router_leg(run_dir, rows, progress):
    """The LLM's half of the router: its prompt with a 78th option, over what ModernBERT was unsure of."""
    import llm
    bert_rows = {r["id"]: r for r in common.read_jsonl(os.path.join(run_dir, "modernbert.jsonl"))}
    unsure = [r for r in rows if bert_rows.get(r["id"]) and (bert_rows[r["id"]].get("confidence") or 0) <= THRESHOLD]
    client = llm.client()
    meta = {"model": "llm_open", "config": client.cfg, "threshold": THRESHOLD,
            "system_prompt": llm.system_prompt(client.cfg["descriptions"], open_set=True)}
    run_model("llm_open", unsure, os.path.join(run_dir, "llm_open.jsonl"),
              lambda text: client.ask(text, open_set=True), os.path.join(run_dir, "llm_open.meta.json"), meta, progress)


def cmd_run(args):
    rows = common.split("test")
    if args.n:
        rows = rows[:args.n]
    run_dir = args.resume or os.path.join(common.RUNS, now_stamp())
    os.makedirs(run_dir, exist_ok=True)
    names = set(args.only.split(",")) if args.only else set(MODELS)
    if names - set(MODELS):
        sys.exit(f"--only takes {', '.join(MODELS)}")
    lock = threading.Lock()
    started = time.time()

    def progress(name, i, n):
        with lock:
            print(f"  {time.time() - started:6.0f}s  {name}: {i}/{n}", flush=True)

    # The run keeps the time it first started; a resume adds a session rather than restamping it.
    run_path = os.path.join(run_dir, "run.json")
    run = json.load(open(run_path, encoding="utf-8")) if os.path.exists(run_path) else {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_local": time.strftime("%Y-%m-%d %H:%M %Z"), "n": len(rows),
        "fern_base": args.fern_base, "host": os.uname().nodename.split(".")[0], "sessions": []}
    run.setdefault("sessions", []).append({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           "models": sorted(names)})
    with open(run_path, "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)
    print(f"run {run_dir}: {len(rows)} messages, models {sorted(names)}")

    if "modernbert" in names:
        run_modernbert(rows, os.path.join(run_dir, "modernbert.jsonl"),
                       os.path.join(run_dir, "modernbert.meta.json"), progress)

    threads = []
    if "llm" in names:
        ask, meta = make_llm()
        threads.append(threading.Thread(target=run_model, args=(
            "llm", rows, os.path.join(run_dir, "llm.jsonl"), ask, os.path.join(run_dir, "llm.meta.json"), meta,
            progress)))
    if "fern" in names:
        ask, meta = make_fern(args.fern_base)
        threads.append(threading.Thread(target=run_model, args=(
            "fern", rows, os.path.join(run_dir, "fern.jsonl"), ask, os.path.join(run_dir, "fern.meta.json"), meta,
            progress)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if "llm" in names and os.path.exists(os.path.join(run_dir, "modernbert.jsonl")):
        run_router_leg(run_dir, rows, progress)
    print(f"done in {time.time() - started:.0f}s")
    if not args.n:
        report(run_dir)


# ── The report ────────────────────────────────────────────────────────────────

def load_model(run_dir, name):
    path = os.path.join(run_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        return None, None
    meta_path = os.path.join(run_dir, f"{name}.meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    if "entry" in meta:  # Fern's model, job and tool-set ids, and the API's address, stay out of the published results
        meta["entry"] = {k: meta["entry"][k] for k in ("base", "records", "hyperparameters") if k in meta["entry"]}
    return {r["id"]: r for r in common.read_jsonl(path)}, meta


def summarise(rows, results, name, meta):
    ok = [results[r["id"]] for r in rows if r["id"] in results and "error" not in results[r["id"]]]
    right = sum(1 for r in rows if r["id"] in results and results[r["id"]].get("label") == r["label"])
    errors = sum(1 for r in rows if r["id"] in results and "error" in results[r["id"]])
    missing = sum(1 for r in rows if r["id"] not in results)
    ms = [x["ms"] for x in ok]
    mean_s = sum(ms) / len(ms) / 1000 if ms else None
    valid = set(common.labels())
    invented = sum(1 for x in ok if x.get("label") not in valid and x.get("label") != "none_of_these")
    out = {"n": len(rows), "right": right, "accuracy": right / len(rows), "errors": errors, "missing": missing,
           "not_an_intent": invented,
           "p50_ms": common.percentile(ms, 50), "p95_ms": common.percentile(ms, 95),
           "mean_ms": mean_s * 1000 if mean_s else None}
    if name == "llm":
        usd = sum(x["cost_usd"] for x in ok)
        sums = {k: sum((x["tokens"].get(k) or 0) for x in ok) for k in ("input", "cache_write", "cache_read", "output", "reasoning")}
        prompt = sums["input"] + sums["cache_write"] + sums["cache_read"]
        out["cost_per_10k_usd"] = usd / len(ok) * 10000
        out["tokens_per_call"] = {k: v / len(ok) for k, v in sums.items()}
        out["tokens_per_call"]["prompt"] = prompt / len(ok)
        out["cached_share"] = sums["cache_read"] / prompt if prompt else 0
        p_ = meta.get("prices", {})
        out["cost_formula"] = (f"per call = (uncached input tokens x {price(p_['input'])} + cache writes x "
                               f"{price(p_['cache_write'])} + cache reads x {price(p_['cache_read'])} + output x "
                               f"{price(p_['output'])}) / 1M, from the API's own usage fields; summed over the run, "
                               "divided by calls, x 10,000")
        served = sorted({x.get("model") for x in ok if x.get("model")})
        out["served_model"] = served
    elif name == "modernbert":
        out["instance"] = INSTANCE
        out["cost_per_10k_usd"] = mean_s * 10000 * INSTANCE["usd_per_hour"] / 3600
        out["cost_formula"] = (f"mean seconds per call ({mean_s:.4f}) x 10,000 x ${INSTANCE['usd_per_hour']}/hour "
                               f"({INSTANCE['instance']}) / 3,600")
    else:
        out["plan"] = PLAN
        out["cost_per_10k_usd"] = None  # a flat fee has no per-call price to set beside the others
        out["cost_formula"] = (f"a flat ${PLAN['usd_per_month']} a month ({PLAN['plan']}, {PLAN['terms']}), "
                               "whatever the volume")
    confs = [x.get("confidence") for x in ok if x.get("confidence") is not None]
    out["has_confidence"] = bool(confs)
    return out


def routing(rows, bert_rows, llm_rows, llm_open_rows):
    """ModernBERT answers above THRESHOLD; the rest go to the LLM. ModernBERT runs on every message either way."""
    sure = [r for r in rows if bert_rows[r["id"]].get("confidence", 0) > THRESHOLD]
    rest = [r for r in rows if bert_rows[r["id"]].get("confidence", 0) <= THRESHOLD]
    bert_right = sum(1 for r in sure if bert_rows[r["id"]]["label"] == r["label"])

    def leg(src):
        if src is None or any(r["id"] not in src for r in rest):
            return None
        right = sum(1 for r in rest if src[r["id"]].get("label") == r["label"])
        usd = sum(src[r["id"]].get("cost_usd", 0) for r in rest)
        return {"right": right, "usd": usd}

    out = {"threshold": THRESHOLD, "modernbert_share": len(sure) / len(rows), "modernbert_handles": len(sure),
           "modernbert_right_on_its_share": bert_right,
           "modernbert_accuracy_on_its_share": bert_right / len(sure) if sure else None,
           "to_llm": len(rest)}
    for kind, src in (("closed", llm_rows), ("open", llm_open_rows)):
        legd = leg(src)
        if not legd:
            continue
        total_right = bert_right + legd["right"]
        out[kind] = {"llm_right_on_rest": legd["right"],
                     "llm_accuracy_on_rest": legd["right"] / len(rest) if rest else None,
                     "blended_right": total_right, "blended_accuracy": total_right / len(rows),
                     "llm_usd_per_10k": legd["usd"] / len(rows) * 10000}
    return out


# Never round in a model's favour: accuracy is rounded down, latency and cost up.
def pct(v):
    return "n/a" if v is None else f"{math.floor(v * 1000 + 1e-9) / 10:.1f}%"


def price(v):
    """A list price as written on the price list: $0.10, $0.125, $0.01."""
    text = f"{v:.3f}".rstrip("0")
    return "$" + (text + "0" if len(text.split(".")[1]) < 2 else text)


def fmt_ms(v):
    return "n/a" if v is None else f"{math.ceil(v - 1e-9):,}"


def fmt_usd(v):
    if v is None:
        return "n/a"
    places = 4 if v < 1 else 2
    return f"${math.ceil(v * 10 ** places - 1e-9) / 10 ** places:,.{places}f}"


def fmt_cost(summary):
    if summary.get("plan"):
        return "N/A"
    return fmt_usd(summary["cost_per_10k_usd"])


def report(run_dir):
    rows = common.split("test")
    run = json.load(open(os.path.join(run_dir, "run.json"), encoding="utf-8"))
    models = {}
    for name in MODELS:
        res, meta = load_model(run_dir, name)
        if res is not None:
            models[name] = {"results": res, "meta": meta, "summary": summarise(rows, res, name, meta)}
    llm_open, _ = load_model(run_dir, "llm_open")
    route = None
    if "modernbert" in models:
        route = routing(rows, models["modernbert"]["results"], models.get("llm", {}).get("results"), llm_open)
        bert_cost = models["modernbert"]["summary"]["cost_per_10k_usd"]
        for kind in ("closed", "open"):
            if route.get(kind):
                route[kind]["blended_usd_per_10k"] = bert_cost + route[kind]["llm_usd_per_10k"]

    data = {"run": os.path.basename(run_dir.rstrip("/")), "started_utc": run["started_utc"],
            "started_local": run["started_local"], "dataset": common.DATASET, "test_messages": len(rows),
            "train_messages": len(common.split("train")),
            "models": {k: {"summary": v["summary"], "meta": v["meta"]} for k, v in models.items()},
            "routing": route}
    with open(os.path.join(common.HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.write("\n")
    with open(os.path.join(common.HERE, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(data))
    # The replay cache for the page: every LLM and Fern answer from this run, keyed by message.
    os.makedirs(CACHE, exist_ok=True)
    for name in ("llm", "fern", "llm_open"):
        src = os.path.join(run_dir, f"{name}.jsonl")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(CACHE, f"{name}.jsonl"))
    with open(os.path.join(CACHE, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"run": data["run"], "started_utc": data["started_utc"]}, fh, indent=2)
    print(f"wrote RESULTS.md, results.json and cache/ from {run_dir}")


def model_name(name, meta):
    if name == "llm":
        return meta.get("config", {}).get("model", "?")
    if name == "modernbert":
        return "ModernBERT-base (149M)"
    return f"{meta.get('base_name', 'Fern')} (`{meta.get('base')}`)"


def markdown(d):
    models = d["models"]
    order = [m for m in MODELS if m in models]
    train_n = {"llm": "0", "modernbert": f"{d['train_messages']:,}", "fern": f"{d['train_messages']:,}"}
    out = [
        "# Results: three ways to route a support message",
        "",
        f"Run **{d['run']}**, started {d['started_utc']} ({d['started_local']}). "
        f"All {d['test_messages']:,} test messages of {d['dataset']['attribution']}, "
        "77 intents, sent one at a time. Written by `bench.py report`; do not edit by hand.",
        "",
        "| | " + " | ".join(TITLES[m] for m in order) + " |",
        "|---|" + "---:|" * len(order),
        "| Model | " + " | ".join(model_name(m, models[m]["meta"]) for m in order) + " |",
        "| Accuracy | " + " | ".join(
            f"**{pct(models[m]['summary']['accuracy'])}** ({models[m]['summary']['right']:,}/{d['test_messages']:,})"
            for m in order) + " |",
        "| p50 latency | " + " | ".join(f"{fmt_ms(models[m]['summary']['p50_ms'])} ms" for m in order) + " |",
        "| p95 latency | " + " | ".join(f"{fmt_ms(models[m]['summary']['p95_ms'])} ms" for m in order) + " |",
        "| Cost per 10,000 calls | " + " | ".join(
            fmt_cost(models[m]["summary"]) + (f" (flat ${PLAN['usd_per_month']} a month, {PLAN['plan']})"
                                              if models[m]["summary"].get("plan") else "")
            for m in order) + " |",
        "| Training examples | " + " | ".join(train_n[m] for m in order) + " |",
        "| Answers that are not one of the 77 | " + " | ".join(
            str(models[m]["summary"].get("not_an_intent", 0)) for m in order) + " |",
        "| Errors (scored wrong) | " + " | ".join(str(models[m]["summary"]["errors"]) for m in order) + " |",
        "",
        "Each model ran on its own, one message at a time: " + "; ".join(
            f"{SHORT[m]} {models[m]['meta'].get('started_utc', '?')} to {models[m]['meta'].get('finished_utc', '?')}"
            for m in order) + " (UTC).",
        "",
        "Nothing is rounded in a model's favour: accuracy is rounded down, latency and cost are rounded up. "
        "The exact counts are beside each accuracy, and every raw answer is in `results.json` and the run directory.",
        "",
    ]
    if "fern" in models:
        out += ["Fern has no price per call to set beside the others: Fernfly charges a flat monthly fee, whatever "
                "the volume.", ""]
    r = d.get("routing")
    if r and r.get("closed"):
        c = r["closed"]
        out += [
            "## Routing: ModernBERT answers when it is sure, the LLM takes the rest",
            "",
            f"ModernBERT answers when its confidence is above {r['threshold']}; everything else goes to the LLM.",
            "",
            "| Share ModernBERT handles | ModernBERT accuracy on its share | LLM accuracy on the rest | "
            "Blended accuracy | Blended cost per 10,000 |",
            "|---:|---:|---:|---:|---:|",
            f"| {r['modernbert_share'] * 100:.1f}% ({r['modernbert_handles']:,}) | "
            f"{pct(r['modernbert_accuracy_on_its_share'])} | "
            f"{pct(c['llm_accuracy_on_rest'])} ({r['to_llm']:,} messages) | "
            f"**{pct(c['blended_accuracy'])}** ({c['blended_right']:,}/{d['test_messages']:,}) | "
            f"{fmt_usd(c['blended_usd_per_10k'])} |",
            "",
            "Blended cost is ModernBERT on every message (it always runs first) plus the LLM on the share passed to it.",
            "",
        ]
        if r.get("open"):
            o = r["open"]
            out += [f"The page's router sends the rest to the LLM with one more option, `none_of_these`, and a "
                    f"free-text reply, which is what lets it handle a message no intent covers. With that prompt the "
                    f"LLM gets {o['llm_right_on_rest']:,}/{r['to_llm']:,} of the rest right (every test message has "
                    f"an intent, so `none_of_these` counts as wrong), for a blended {pct(o['blended_accuracy'])} at "
                    f"{fmt_usd(o['blended_usd_per_10k'])} per 10,000.", ""]
    oos_path = os.path.join(common.HERE, "out-of-set.json")
    if os.path.exists(oos_path):
        oos = json.load(open(oos_path, encoding="utf-8"))
        n_oos = len(oos["messages"])
        out += ["## Requests that fit none of the 77", "",
                f"{n_oos} banking requests we wrote (`out_of_set.py`). ModernBERT has no \"none of these\" label, so "
                f"the router's only defence is its confidence. {oos['routed']} of {n_oos} fell to {oos['threshold']} "
                f"or below and went to the LLM. {n_oos - oos['routed']} did not: ModernBERT was sure, and wrong.", ""]
        for r in oos["modernbert_confident_and_wrong"]:
            out.append(f"- \"{r['text']}\": ModernBERT said `{r['modernbert_label']}` at {r['modernbert_confidence']:.2f}.")
        asked = [r for r in oos["messages"] if r.get("llm_label")]
        if asked:
            none = sum(1 for r in asked if r["llm_label"] == "none_of_these")
            out += ["", f"With the router's open prompt (the LLM's, plus `none_of_these` and a reply), the LLM "
                    f"answered `none_of_these` on {none} of the {len(asked)} it was sent, and wrote a reply to each. "
                    "Some replies state bank policy the prompt never gave it (\"we don't offer joint accounts\"): the "
                    "open-ended path is the LLM's to handle, but in a product it needs the bank's real policies to "
                    "answer from."]
        out.append("")
    pick_path = os.path.join(common.HERE, "llm-pick.json")
    if os.path.exists(pick_path) and "llm" in models:
        pick = json.load(open(pick_path, encoding="utf-8"))
        out += ["## Which LLM setup", "",
                f"The LLM's setup was chosen on {pick['n']} training messages (never test ones), the same "
                f"{pick['n']} for each candidate. The chosen one is in bold.", "",
                "| Setup | Right | p50 | p95 | Cost per 10,000 |", "|---|---:|---:|---:|---:|"]
        for r in pick["candidates"]:
            label = r["label"]
            if r.get("chosen"):
                label = f"**{label}**"
            out.append(f"| {label} | {r['right']}/{r['n']} | {fmt_ms(r['p50_ms'])} ms | {fmt_ms(r['p95_ms'])} ms | "
                       f"{fmt_usd(r['usd_per_10k'])} |")
        out += ["", pick.get("why", ""), ""]
    out += ["## How each number was measured", ""]
    for name in order:
        s, m = models[name]["summary"], models[name]["meta"]
        out.append(f"### {TITLES[name]}")
        out.append("")
        if name == "llm":
            cfg = m.get("config", {})
            if cfg.get("provider") == "anthropic":
                knobs = (f"thinking {cfg.get('thinking')}, effort {cfg.get('effort')}, structured outputs "
                         "(`output_config.format`) with the 77 intents as an enum, the intent list cached with "
                         "`cache_control`. The Claude API returns no token probabilities, so the LLM has no confidence")
            else:
                knobs = (f"temperature {cfg.get('temperature')}, reasoning effort {cfg.get('reasoning_effort') or 'not set'}, "
                         "structured outputs with the 77 intents as an enum, `prompt_cache_key`, confidence from logprobs")
            out.append(f"- Model `{cfg.get('model')}` (served as {', '.join(s.get('served_model') or ['?'])}): {knobs}. "
                       f"Prompt `{m.get('prompt_version')}` "
                       f"({'names and one-line descriptions' if cfg.get('descriptions') else 'names only'}).")
            t = s["tokens_per_call"]
            out.append(f"- Tokens per call, from the API's usage fields: {t['prompt']:,.0f} input "
                       f"({s['cached_share'] * 100:.0f}% of them read from the cache), {t['output']:.1f} output.")
            p = m.get("prices", {})
            out.append(f"- List price on {m.get('prices_as_of')}, per 1M tokens: {price(p['input'])} input, "
                       f"{price(p['cache_write'])} cache write, {price(p['cache_read'])} cache read, "
                       f"{price(p['output'])} output ({m.get('prices_url')}).")
        elif name == "modernbert":
            ck = m.get("checkpoint", {})
            out.append(f"- `{ck.get('base')}` at revision `{str(ck.get('base_revision'))[:12]}`, fine-tuned for "
                       f"{ck.get('recipe', {}).get('epochs')} epochs ({ck.get('steps')} steps) on {ck.get('train_examples'):,} "
                       f"messages in {ck.get('train_seconds', 0) / 60:.0f} minutes on {ck.get('device')} "
                       f"(torch {ck.get('torch')}). No early stopping, nothing tuned on the test set.")
            out.append(f"- Run on a laptop ({m.get('device')}), one message at a time, in-process: no network hop, "
                       "because ModernBERT runs where the app runs.")
        else:
            e = m.get("entry", {})
            out.append(f"- {m.get('base_name', 'Fern')} (`{e.get('base')}`), fine-tuned on "
                       f"{e.get('records', 0):,} messages, hyperparameters {json.dumps(e.get('hyperparameters'))} "
                       "(Fernfly's defaults for this many pairs). One tool per intent, no arguments.")
            if e.get("base") == "fern-1500":
                out.append("- Fern Pinna learns its tools in training, so at inference it is sent only the message.")
            out.append("- Served from one GPU in UK South, so every latency includes a round trip from the laptop "
                       "to the UK and back.")
            out.append(f"- Fern is generative, so it can name a tool it was never given: it did on "
                       f"{s.get('not_an_intent', 0)} of the {s['n']:,} messages (scored wrong). The LLM cannot, "
                       "because structured outputs hold it to the 77; ModernBERT cannot, because it is a classifier "
                       "over the 77.")
        if s.get("plan"):
            plan = s["plan"]
            out.append(f"- Cost per 10,000 calls: N/A. Fernfly charges {s['cost_formula']} ({plan['source']}, "
                       f"{plan['as_of']}): not by compute, tokens or calls.")
        else:
            out.append(f"- Cost per 10,000 calls: {s['cost_formula']} = **{fmt_usd(s['cost_per_10k_usd'])}**.")
        if s.get("instance"):
            inst = s["instance"]
            out.append(f"- {inst['instance']}: {inst['chip']}, ${inst['usd_per_hour']}/hour on demand, us-east-1, "
                       f"{inst['source']} (2026-11-23). {inst['note']}")
        out.append(f"- p50 {fmt_ms(s['p50_ms'])} ms, p95 {fmt_ms(s['p95_ms'])} ms, mean {fmt_ms(s['mean_ms'])} ms "
                   f"(nearest-rank percentiles), {s['errors']} errors, {s['missing']} not run.")
        out.append("")
    out += ["## Data", "",
            f"{d['dataset']['attribution']}: {d['dataset']['url']} at revision `{d['dataset']['revision'][:12]}`. "
            f"{d['train_messages']:,} training and {d['test_messages']:,} test messages.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the three models on Banking77")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--only", help="llm, modernbert, fern, or a comma-separated list")
    p.add_argument("--resume")
    p.add_argument("--n", type=int, help="first N test messages only (a smoke test; writes no report)")
    p.add_argument("--fern-base", default="fern-1500")
    sub.add_parser("report").add_argument("run_dir")
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        report(args.run_dir)


if __name__ == "__main__":
    main()
