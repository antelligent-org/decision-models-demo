#!/usr/bin/env python3
"""
serve.py: the demo's service, and the page it serves.

    .venv/bin/python serve.py              # http://127.0.0.1:8765
    .venv/bin/python serve.py --cached     # start in replay mode: no network needed
    .venv/bin/python serve.py --no-bert    # without torch: the LLM and Fern (or their cache) only

It loads ModernBERT on this machine, holds one kept-alive connection each to the
LLM's API and the Fern API, and answers the page one message and one model at a
time, so each model shows its own latency. A model that is down (no network, an
expired key) is retried every 30 seconds in the background, so it comes back by
itself.

Every LLM and Fern answer, from the benchmark run (cache/llm.jsonl,
cache/fern.jsonl) and from every use of the page (cache/live.jsonl), is kept by
message text. When a live call fails, or the page is switched to cached mode
(key C), the answer comes from there instead, after waiting as long as the
recorded call took, and the page labels it "cached". ModernBERT never needs the
network.

Binds to 127.0.0.1 only. Never prints a key.
"""

import argparse
import copy
import json
import mimetypes
import os
import random
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

WEB = os.path.join(common.HERE, "web")
CACHE = os.path.join(common.HERE, "cache")
LIVE_CACHE = os.path.join(CACHE, "live.jsonl")
DEMO_FILE = os.path.join(common.HERE, "demo.json")  # the messages each view uses
THRESHOLD = 0.9
RETRY_SECONDS = 30
MODELS = ("llm", "modernbert", "fern")
REMOTE = ("llm", "llm_open", "fern")  # llm_open: the router's LLM, with "none of these" and a reply
NAMES = {"llm": "LLM", "llm_open": "LLM", "modernbert": "ModernBERT", "fern": "Fern"}


class State:
    def __init__(self, args):
        self.mode = "cached" if args.cached else "live"
        self.lock_bert = threading.Lock()
        self.lock_cache = threading.Lock()
        self.locks = {name: threading.Lock() for name in REMOTE}
        self.test = common.split("test")
        self.by_text = {r["text"]: r for r in self.test}
        self.train = common.split("train")
        self.cache = {name: {} for name in REMOTE}
        self.errors = {}
        self.load_cache()
        self.bert_model = self.bert_original = self.tok = None
        self.bert_info = None
        self.retrain = {"state": "idle"}
        if not args.no_bert:
            self.load_modernbert()
        self.clients = {name: None for name in REMOTE}
        self.llm_model = None
        self.fern_base = args.fern_base
        self.connect()
        threading.Thread(target=self._keep_connected, daemon=True).start()

    # ── ModernBERT ──
    def load_modernbert(self):
        import bert
        self.bert = bert
        self.bert_model, self.tok = bert.load()
        self.bert_original = self.bert_model
        self.bert_info = bert.meta()
        for _ in range(10):
            bert.ask_one(self.bert_model, self.tok, "warming up the model")
        # ModernBERT's answers on the whole test set, for the retrain view's before and after.
        # Kept beside the checkpoint so the view does not pay for them each time.
        path = os.path.join(bert.CHECKPOINT, "test-predictions.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.baseline = json.load(fh)
        else:
            print("  scoring ModernBERT on the test set once, for the retrain view ...", flush=True)
            self.baseline = bert.predict(self.bert_model, self.tok, [x["text"] for x in self.test])
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.baseline, fh)

    def ask_modernbert(self, text):
        if not self.bert_model:
            raise RuntimeError("ModernBERT is not loaded (started with --no-bert)")
        with self.lock_bert:
            out = self.bert.ask_one(self.bert_model, self.tok, text)
        return {"label": out["label"], "confidence": out["confidence"], "ms": out["ms"], "top": out["top"],
                "retrained": self.bert_model is not self.bert_original}

    # ── The LLM and Fern ──
    def connect(self):
        self.connect_llm()
        self.connect_fern()

    def connect_llm(self):
        try:
            import llm
            closed = llm.client()
            self.llm_model = closed.cfg["model"]
            closed.ask("warming up")
            open_ = llm.client()  # its own connection, so the router never queues behind the LLM column
            open_.ask("warming up", open_set=True)
            self.clients.update(llm=closed, llm_open=open_)
            self.errors.pop("llm", None)
        except BaseException as err:  # noqa: BLE001  (SystemExit when there is no key)
            self.errors["llm"] = str(err)[:300]
            self.clients.update(llm=None, llm_open=None)

    def connect_fern(self):
        try:
            import fern
            client = fern.Client(self.fern_base)
            client.ask("warming up")
            self.clients["fern"] = client
            self.errors.pop("fern", None)
        except BaseException as err:  # noqa: BLE001
            self.errors["fern"] = str(err)[:300]
            self.clients["fern"] = None

    def _keep_connected(self):
        """Retry a model that is down, so a dropped network or a renewed key heals without a restart."""
        while True:
            time.sleep(RETRY_SECONDS)
            if self.mode != "live":
                continue
            if self.clients["llm"] is None:
                self.connect_llm()
            if self.clients["fern"] is None:
                self.connect_fern()

    def load_cache(self):
        for name in REMOTE:
            path = os.path.join(CACHE, f"{name}.jsonl")
            if os.path.exists(path):
                rows = {r["id"]: r for r in common.read_jsonl(path)}
                for r in self.test:
                    if r["id"] in rows and "error" not in rows[r["id"]]:
                        self.cache[name][r["text"]] = rows[r["id"]]
        if os.path.exists(LIVE_CACHE):
            for row in common.read_jsonl(LIVE_CACHE):
                self.cache.setdefault(row["model"], {})[row["text"]] = row["answer"]

    def remember(self, name, text, answer):
        keep = {k: v for k, v in answer.items() if k != "raw"}
        with self.lock_cache:
            self.cache.setdefault(name, {})[text] = keep
            os.makedirs(CACHE, exist_ok=True)
            with open(LIVE_CACHE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": name, "text": text, "answer": keep, "at": time.time()},
                                    ensure_ascii=False) + "\n")

    def replay(self, name, text, why):
        hit = self.cache.get(name, {}).get(text)
        if not hit:
            raise RuntimeError(f"{why}; no cached answer for this message")
        time.sleep(min(hit.get("ms", 0), 5000) / 1000)
        out = dict(hit, cached=True, cached_why=why)
        out.pop("id", None)
        return out

    def ask_remote(self, name, text):
        client = self.clients[name]
        if self.mode == "cached":
            return self.replay(name, text, "cached mode")
        if client is None:
            return self.replay(name, text, self.errors.get("llm" if name == "llm_open" else name, "not connected"))
        try:
            with self.locks[name]:  # one kept-alive connection per client: one call at a time on it
                out = client.ask(text, open_set=True) if name == "llm_open" else client.ask(text)
        except Exception as err:  # noqa: BLE001
            return self.replay(name, text, f"live call failed: {str(err)[:120]}")
        out.pop("raw", None)
        self.remember(name, text, out)
        return dict(out, cached=False)

    def ask(self, name, text):
        out = self.ask_modernbert(text) if name == "modernbert" else self.ask_remote(name, text)
        gold = self.by_text.get(text, {}).get("label")
        out["gold"] = gold
        out["correct"] = None if gold is None else out.get("label") == gold
        return out

    # ── The retrain view ──
    def start_retrain(self, text, label):
        if self.retrain.get("state") == "running":
            raise RuntimeError("a retrain is already running")
        if not self.bert_model:
            raise RuntimeError("ModernBERT is not loaded")
        self.retrain = {"state": "running", "phase": "copying the model", "text": text, "label": label,
                        "started": time.time(), "step": 0, "steps": self.bert.RETRAIN["steps"]}
        threading.Thread(target=self._retrain, args=(text, label), daemon=True).start()

    def _retrain(self, text, label):
        r = self.retrain
        try:
            bert = self.bert
            with self.lock_bert:
                before_one = bert.predict(self.bert_model, self.tok, [text], batch_size=1)[0]
            r["before"] = before_one
            model = copy.deepcopy(self.bert_original)
            r["phase"] = "training"
            t0 = time.time()

            def on_step(i, n, loss):
                r.update(step=i, steps=n, loss=loss)

            bert.retrain_one(model, self.tok, text, label, self.train, on_step=on_step)
            r["train_seconds"] = time.time() - t0
            r["after"] = bert.predict(model, self.tok, [text], batch_size=1)[0]
            r["phase"] = "rerunning all 3,080 test messages"
            t1 = time.time()
            after = bert.predict(model, self.tok, [x["text"] for x in self.test])
            r["rerun_seconds"] = time.time() - t1
            gold = [x["label"] for x in self.test]
            was = [p["label"] == g for p, g in zip(self.baseline, gold)]
            now = [p["label"] == g for p, g in zip(after, gold)]
            r["accuracy_before"] = sum(was) / len(gold)
            r["accuracy_after"] = sum(now) / len(gold)
            r["right_before"], r["right_after"] = sum(was), sum(now)
            r["fixed"] = [{"text": self.test[i]["text"], "gold": gold[i], "was": self.baseline[i]["label"]}
                          for i in range(len(gold)) if now[i] and not was[i]]
            r["broken"] = [{"text": self.test[i]["text"], "gold": gold[i], "now": after[i]["label"]}
                           for i in range(len(gold)) if was[i] and not now[i]]
            with self.lock_bert:
                self.bert_model = model
            r["phase"] = "done"
            r["state"] = "done"
            r["seconds"] = time.time() - r["started"]
        except Exception as err:  # noqa: BLE001
            traceback.print_exc()
            r.update(state="failed", error=str(err)[:300])

    def reset_modernbert(self):
        with self.lock_bert:
            self.bert_model = self.bert_original
        self.retrain = {"state": "idle"}

    def info(self):
        results = None
        path = os.path.join(common.HERE, "results.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                results = json.load(fh)
        demo = {}
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, encoding="utf-8") as fh:
                demo = json.load(fh)
        fern_client = self.clients["fern"]
        fern_entry = fern_client.entry if fern_client else None
        fern_size = {}
        try:
            import fern
            fern_entry = fern_entry or fern.models().get(f"banking77@{self.fern_base}")
            fern_size = fern.BASES.get(self.fern_base, {})
        except Exception:  # noqa: BLE001
            pass
        llm_client = self.clients["llm"]
        return {
            "mode": self.mode, "threshold": THRESHOLD, "dataset": common.DATASET,
            "intents": [r["label"] for r in common.intents()],
            "train_count": len(self.train), "test_count": len(self.test),
            "llm": {"ready": llm_client is not None, "model": self.llm_model,
                    "provider": llm_client.cfg["provider"] if llm_client else _llm_provider(),
                    "error": self.errors.get("llm"), "cached": len(self.cache.get("llm", {}))},
            "modernbert": {"ready": self.bert_model is not None,
                           "retrained": self.bert_model is not self.bert_original,
                           "checkpoint": self.bert_info},
            "fern": {"ready": fern_client is not None, "base": self.fern_base, "entry": fern_entry,
                     "name": fern_size.get("name"), "params": fern_size.get("params"),
                     "error": self.errors.get("fern"), "cached": len(self.cache.get("fern", {}))},
            "results": results, "demo": demo,
        }


def _llm_provider():
    try:
        import llm
        return llm.config()["provider"]
    except Exception:  # noqa: BLE001
        return None


def sample(state, n, seed):
    rng = random.Random(seed)
    return rng.sample(state.test, n)


class Handler(BaseHTTPRequestHandler):
    server_version = "decision-models/1.0"

    def log_message(self, fmt, *args):  # quieter than the default, and never logs bodies
        if "/api/" in self.path and not self.path.startswith("/api/retrain/status"):
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path.split('?')[0]}\n")

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = urlsplit(self.path).path
        query = dict(p.split("=", 1) for p in urlsplit(self.path).query.split("&") if "=" in p)
        st = self.server.state
        if path == "/api/info":
            return self.send_json(st.info())
        if path == "/api/sample":
            return self.send_json(sample(st, int(query.get("n", 20)), int(query.get("seed", 2611))))
        if path == "/api/retrain/status":
            r = dict(st.retrain)
            if r.get("started") and r.get("state") == "running":
                r["elapsed"] = time.time() - r["started"]
            return self.send_json(r)
        return self.static(path)

    def do_POST(self):
        path = urlsplit(self.path).path
        st = self.server.state
        try:
            body = self.body()
            if path == "/api/ask":
                if body.get("model") not in MODELS:
                    return self.send_json({"error": "model must be one of " + ", ".join(MODELS)}, 400)
                return self.send_json(st.ask(body["model"], body["text"]))
            if path == "/api/warm":  # one unscored ModernBERT call, so an idle GPU is awake before a timed run
                if st.bert_model:
                    st.ask_modernbert("warming up the model")
                return self.send_json({"ok": True})
            if path == "/api/route":
                bert_answer = st.ask("modernbert", body["text"])
                out = {"modernbert": bert_answer, "threshold": THRESHOLD,
                       "routed": (bert_answer.get("confidence") or 0) <= THRESHOLD}
                if out["routed"]:
                    try:
                        out["llm"] = st.ask("llm_open", body["text"])
                    except Exception as err:  # noqa: BLE001
                        out["llm"] = {"error": str(err)[:300]}
                return self.send_json(out)
            if path == "/api/retrain":
                st.start_retrain(body["text"], body["label"])
                return self.send_json({"ok": True})
            if path == "/api/reset":
                st.reset_modernbert()
                return self.send_json({"ok": True})
            if path == "/api/mode":
                st.mode = "cached" if body.get("mode") == "cached" else "live"
                return self.send_json({"mode": st.mode})
            if path == "/api/reconnect":
                st.errors.clear()
                st.connect()
                return self.send_json(st.info())
        except Exception as err:  # noqa: BLE001
            return self.send_json({"error": str(err)[:300]}, 500)
        return self.send_json({"error": "not found"}, 404)

    def static(self, path):
        """Files under web/ only, with web/ as the site root."""
        web = os.path.realpath(WEB)
        full = os.path.realpath(os.path.join(web, path.lstrip("/") or "index.html"))
        if not full.startswith(web + os.sep) or not os.path.isfile(full):
            self.send_response(404)
            self.end_headers()
            return
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(full)[0] or "application/octet-stream")
                         + ("; charset=utf-8" if full.endswith((".html", ".css", ".js")) else ""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="The decision models demo: the service and its page")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cached", action="store_true", help="start in replay mode")
    parser.add_argument("--no-bert", action="store_true", help="do not load torch or ModernBERT")
    parser.add_argument("--fern-base", default=os.environ.get("FERN_BASE", "fern-1500"))
    args = parser.parse_args()
    print("loading ModernBERT and connecting the LLM and Fern ...", flush=True)
    state = State(args)
    for name in ("llm", "fern"):
        if state.errors.get(name):
            print(f"  {NAMES[name]}: not live ({state.errors[name][:120]}); cached answers: {len(state.cache[name])}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.state = state
    print(f"ready: http://127.0.0.1:{args.port}/  (mode {state.mode})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
