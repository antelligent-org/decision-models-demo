#!/usr/bin/env python3
"""
fern.py: the Fern column, a Fern model fine-tuned on the same 10,003 messages.

    python3 fern.py export                    # the two files a Fernfly project needs, in export/
    python3 fern.py ask "my card hasn't arrived" [--base fern-1500]

With Fern API access (FERN_BASE_URL and FERN_API_KEY), it can also train directly:

    python3 fern.py probe                     # register the tools and upload 5 lines, to check the format
    python3 fern.py train --base fern-1500    # or fern-bud-1.0 (Fern Bud), fern-160m (Fern Pinnule)
    python3 fern.py status ftjob-...

How 77 intents become tool calls: one tool per intent, each with no arguments,
so the model's whole decision is which tool to name. Fern models are trained
first and foremost to pick a tool (their published tool-name accuracy is well
above their argument accuracy), and the gateway's post-processing grounds
argument values in the user's words, which suits a date or a name and does not
suit an enum value the user never typed. A tool with an `intent` enum argument
would turn the one thing that matters into an argument to fill.

The steps are the Fern API's own: register a tool set, upload JSONL, queue a
job, poll. The hyperparameters are Fernfly's defaults for 10,003 pairs: what a
customer would get without touching the advanced settings.

Trained models are recorded in fern-models.json, which git ignores.
The address is FERN_BASE_URL and the key FERN_API_KEY, both in .env.local (see
common.secret).
Standard library only.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODELS_FILE = os.path.join(common.HERE, "fern-models.json")
UA = "decision-models/1.0"  # Cloudflare in front of the API refuses urllib's default (error 1010)

# Fernfly's defaults for n = 10,003 pairs.
BASES = {
    "fern-bud-1.0": {"name": "Fern Bud", "params": "12M", "hyper": {"n_epochs": 3, "batch_size": 32, "learning_rate": 1e-4}},
    "fern-160m": {"name": "Fern Pinnule", "params": "160M", "hyper": {"n_epochs": 3, "batch_size": 32, "learning_rate": 1e-4}},
    "fern-1500": {"name": "Fern Pinna", "params": "1.5B", "hyper": {"n_epochs": 2, "batch_size": 16, "learning_rate": 2e-4}},
}

def token():
    """The Fern API key. Read once per call site, never printed."""
    return common.secret("Fern API key", "FERN_API_KEY")


def base_url():
    """The Fern API's address. Not in this repository: set FERN_BASE_URL in .env.local."""
    return common.secret("Fern API address", "FERN_BASE_URL").rstrip("/")


def request(method, path, body=None, content_type="application/json"):
    data = body
    if body is not None and content_type == "application/json":
        data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {token()}", "User-Agent": UA}
    if data is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(base_url() + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as res:
            return json.loads(res.read().decode() or "{}")
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")[:2000]
        sys.exit(f"{method} {path} returned {err.code}: {detail}")


def upload(name, content):
    boundary = uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nfine-tune\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
         f"Content-Type: application/jsonl\r\n\r\n").encode() + content + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return request("POST", "/v1/files", b"".join(parts), f"multipart/form-data; boundary={boundary}")


# ── Converting ────────────────────────────────────────────────────────────────

def tools():
    return [{"type": "function", "function": {"name": row["tool"], "description": row["description"],
                                              "parameters": {"type": "object", "properties": {}}}}
            for row in common.intents()]


def record(text, label, tool_set_id, label_to_tool):
    return {"messages": [{"role": "user", "content": text}], "tool_set_id": tool_set_id,
            "expected_tool_calls": [{"name": label_to_tool[label], "arguments": {}}]}


def jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def register():
    result = request("POST", "/v1/tool_sets", {"tools": tools()})
    tsid = result.get("id") or result.get("tool_set_id")
    if not tsid:
        sys.exit(f"tool set registered, but no id came back: {json.dumps(result)[:500]}")
    return tsid


def training_rows(tsid, extra=()):
    l2t = {row["label"]: row["tool"] for row in common.intents()}
    rows = [(r["text"], r["label"]) for r in common.split("train")] + list(extra)
    return [record(t, lab, tsid, l2t) for t, lab in rows]


def models():
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_model(key, entry):
    data = models()
    data[key] = entry
    with open(MODELS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


# ── Inference ─────────────────────────────────────────────────────────────────

class Client:
    """Asks one fine-tuned Fern model, over one kept-alive connection."""

    def __init__(self, base="fern-bud-1.0", key=None):
        entry = models().get(key or f"banking77@{base}")
        if not entry:
            sys.exit(f"no trained model for {base} in {os.path.basename(MODELS_FILE)}: run fern.py train --base {base}")
        self.entry, self.base = entry, entry["base"]
        self.http = common.Keepalive(base_url() + "/v1",
                                     {"Authorization": f"Bearer {token()}", "User-Agent": UA}, timeout=60)
        self.t2l = {row["tool"]: row["label"] for row in common.intents()}

    def body(self, text):
        body = {"model": self.entry["model"], "messages": [{"role": "user", "content": text}], "temperature": 0}
        if self.base != "fern-1500":  # Pinna learns its tools in training and is sent none
            body["tool_set_id"] = self.entry["tool_set_id"]
        return body

    def ask(self, text):
        status, res, headers, ms = self.http.post("/chat/completions", self.body(text))
        if status != 200:
            raise RuntimeError(f"Fern {status}: {str(res)[:300]}")
        msg = (res.get("choices") or [{}])[0].get("message", {})
        calls = msg.get("tool_calls") or []
        name = calls[0]["function"]["name"] if calls else None
        return {"label": self.t2l.get(name, name or "(no call)"), "confidence": _confidence(res, calls),
                "ms": ms, "model": self.entry["model"], "server_ms": _server_ms(res, headers),
                "raw": {"tool_calls": calls, "content": msg.get("content"),
                        "variant": headers.get("x-variant-used") or headers.get("X-Variant-Used")}}


def _confidence(res, calls):
    """Whatever score the server returns for the call, if any. The tool-callers do not promise one."""
    for source in (calls[0] if calls else {}, res.get("choices", [{}])[0], res):
        for key in ("confidence", "score", "probability"):
            if isinstance(source.get(key), (int, float)):
                return float(source[key])
    return None


def _server_ms(res, headers):
    for key in ("x-process-time", "X-Process-Time", "x-response-time", "server-timing"):
        if headers.get(key):
            return headers[key]
    return (res.get("usage") or {}).get("latency_ms") or res.get("latency_ms")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_export(args):
    """The two files a Fernfly project needs: its tools, and its training pairs.

    banking77.openapi.json  one operation per intent, no parameters: create the project from this spec
    banking77.pairs.jsonl   the 10,003 training messages in Fernfly's import format (Training data, Import)
    """
    out = args.out or os.path.join(common.HERE, "export")
    os.makedirs(out, exist_ok=True)
    spec = {"openapi": "3.0.3",
            "info": {"title": "Banking77 intents", "version": "1.0",
                     "description": "One operation per Banking77 intent, with no parameters: the call is the decision."},
            "paths": {f"/intents/{row['tool']}": {"post": {"operationId": row["tool"], "summary": row["description"],
                                                         "responses": {"200": {"description": "Routed"}}}}
                      for row in common.intents()}}
    with open(os.path.join(out, "banking77.openapi.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    l2t = {row["label"]: row["tool"] for row in common.intents()}
    pairs = [{"input_text": r["text"], "output_json": [{"name": l2t[r["label"]], "arguments": {}}], "approved": 1}
             for r in common.split("train")]
    with open(os.path.join(out, "banking77.pairs.jsonl"), "wb") as fh:
        fh.write(jsonl(pairs))
    print(f"{len(spec['paths'])} tools -> {out}/banking77.openapi.json")
    print(f"{len(pairs)} training pairs -> {out}/banking77.pairs.jsonl")


def cmd_probe(_args):
    tsid = register()
    print(f"tool set  {tsid}  ({len(tools())} tools)")
    rows = training_rows(tsid)
    picks = [rows[i] for i in (0, 2000, 4000, 6000, 8000)]
    print(json.dumps(upload("banking77-probe.jsonl", jsonl(picks)), indent=2)[:3000])


def cmd_train(args):
    tsid = register()
    rows = training_rows(tsid)
    print(f"tool set  {tsid}  ({len(tools())} tools, one per intent, no arguments)")
    uploaded = upload(f"banking77-{args.base}.jsonl", jsonl(rows))
    file_id = uploaded.get("id")
    print(f"file      {file_id}  ({len(rows)} records)")
    if not file_id:
        sys.exit(json.dumps(uploaded)[:2000])
    hyper = dict(BASES[args.base]["hyper"], **({"n_epochs": args.epochs} if args.epochs else {}))
    job = request("POST", "/v1/fine_tuning/jobs", {"training_file": file_id, "model": args.base, "hyperparameters": hyper})
    print(f"job       {job.get('id')}  {args.base} {hyper}", flush=True)
    started = time.time()
    job = wait(job["id"], args.timeout)
    if job.get("status") != "succeeded":
        sys.exit(f"job ended {job.get('status')}: {job.get('error')}")
    save_model(f"banking77@{args.base}", {
        "model": job["fine_tuned_model"], "base": args.base, "tool_set_id": tsid, "job": job["id"],
        "records": len(rows), "hyperparameters": hyper,
        "final_loss": job.get("final_loss"), "trained_steps": job.get("trained_steps"),
        "created_at": job.get("created_at"), "finished_at": job.get("finished_at"),
        "wall_seconds_waited": round(time.time() - started)})
    print(f"model     {job['fine_tuned_model']}  (recorded in {os.path.basename(MODELS_FILE)})")


def wait(job_id, timeout):
    start, last = time.time(), None
    while True:
        job = request("GET", f"/v1/fine_tuning/jobs/{job_id}")
        line = f"{job.get('status')}  steps {job.get('trained_steps')}  loss {job.get('loss')}"
        if line != last:
            print(f"  {int(time.time() - start):>5}s  {line}", flush=True)
            last = line
        if job.get("status") in ("succeeded", "failed", "cancelled", "stopped"):
            return job
        if time.time() - start > timeout:
            sys.exit(f"still {job.get('status')} after {timeout}s; check later with: python3 fern.py status {job_id}")
        time.sleep(10)


def cmd_status(args):
    print(json.dumps(request("GET", f"/v1/fine_tuning/jobs/{args.job}"), indent=2))


def cmd_ask(args):
    client = Client(args.base)
    client.ask("warm up")
    print(json.dumps(client.ask(" ".join(args.text)), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train and ask the Banking77 Fern model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export").add_argument("--out")
    sub.add_parser("probe")
    p = sub.add_parser("train")
    p.add_argument("--base", choices=tuple(BASES), default="fern-bud-1.0")
    p.add_argument("--epochs", type=int)
    p.add_argument("--timeout", type=int, default=4 * 3600)
    sub.add_parser("status").add_argument("job")
    p = sub.add_parser("ask")
    p.add_argument("text", nargs="+")
    p.add_argument("--base", choices=tuple(BASES), default="fern-bud-1.0")
    args = parser.parse_args()
    {"export": cmd_export, "probe": cmd_probe, "train": cmd_train, "status": cmd_status,
     "ask": cmd_ask}[args.command](args)


if __name__ == "__main__":
    main()
