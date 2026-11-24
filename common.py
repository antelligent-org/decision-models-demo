"""
common.py: the data, the labels, and the prices, shared by every script here.

Standard library only, so the service and the page can run without torch when
only the cached replay is needed.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")      # gitignored: the ModernBERT checkpoints
RUNS = os.path.join(HERE, "runs")          # one directory per benchmark run

DATASET = {
    "name": "Banking77",
    "hf_id": "PolyAI/banking77",
    "revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
    "license": "CC BY 4.0",
    "attribution": "Banking77 by PolyAI (Casanueva et al., 2020), CC BY 4.0",
    "url": "https://huggingface.co/datasets/PolyAI/banking77",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def labels():
    """The 77 intent names, in the dataset's own index order."""
    with open(os.path.join(DATA, "labels.json"), encoding="utf-8") as fh:
        return json.load(fh)


def split(name):
    """'train' or 'test': a list of {"id", "text", "label"} with label as the name."""
    return read_jsonl(os.path.join(DATA, f"{name}.jsonl"))


ENV_LOCAL = os.path.join(HERE, ".env.local")  # every key lives here, and git ignores it


def env_local(var):
    """One VAR=value from .env.local, or None. The file is read, never printed."""
    if not os.path.exists(ENV_LOCAL):
        return None
    with open(ENV_LOCAL, encoding="utf-8") as fh:
        for line in fh:
            key, sep, value = line.strip().partition("=")
            if sep and key.strip().removeprefix("export ").strip() == var:
                return value.strip().strip("\"'") or None
    return None


def secret(name, var):
    """A key from .env.local, never printed. Only from there: other tools set variables
    such as ANTHROPIC_API_KEY for themselves, and one place for every key is easier to keep.
    """
    value = (env_local(var) or "").strip()
    if not value:
        raise SystemExit(f"no {name}: put {var}=... in {ENV_LOCAL}")
    return value


def intents():
    """[{label, tool, description}] for the 77 intents, in dataset order."""
    with open(os.path.join(DATA, "intents.json"), encoding="utf-8") as fh:
        rows = json.load(fh)["intents"]
    if [r["label"] for r in rows] != labels():
        raise SystemExit("data/intents.json is out of step with data/labels.json")
    return rows


class Keepalive:
    """One persistent HTTPS connection to one host, as a production client would hold.

    A fresh TLS handshake on every call would add a round trip or two to every
    latency we measure, for the LLM and Fern alike. Reconnects once on a dropped
    connection. Not thread-safe: give each thread its own.
    """

    def __init__(self, base_url, headers, timeout=60):
        from urllib.parse import urlsplit
        parts = urlsplit(base_url)
        self.host, self.prefix = parts.netloc, parts.path.rstrip("/")
        self.headers = dict(headers, **{"Content-Type": "application/json"})
        self.timeout, self.conn = timeout, None

    def _connect(self):
        import http.client
        self.conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)

    def post(self, path, body, extra_headers=None):
        """(status, parsed JSON or text, response headers, milliseconds) for one POST."""
        import http.client
        import time
        data = json.dumps(body).encode()
        headers = dict(self.headers, **(extra_headers or {}))
        for attempt in (1, 2):
            if self.conn is None:
                self._connect()
            start = time.perf_counter()
            try:
                self.conn.request("POST", self.prefix + path, data, headers)
                res = self.conn.getresponse()
                raw = res.read()
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                if attempt == 2:
                    raise
                continue
            ms = (time.perf_counter() - start) * 1000
            text = raw.decode(errors="replace")
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = text
            return res.status, parsed, dict(res.getheaders()), ms

    def warm(self, path="/"):
        """Open the connection (TCP and TLS) before the first timed call."""
        import http.client
        if self.conn is None:
            self._connect()
        try:
            self.conn.request("HEAD", self.prefix + path, headers=self.headers)
            self.conn.getresponse().read()
        except (http.client.HTTPException, OSError):
            self.conn.close()
            self.conn = None


def percentile(values, q):
    """Nearest-rank percentile, so p95 is a latency that actually happened."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(-(-q * len(ordered) // 100)))  # ceil(q/100 * n)
    return ordered[min(rank, len(ordered)) - 1]
