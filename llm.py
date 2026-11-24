#!/usr/bin/env python3
"""
llm.py: the LLM, a small hosted model routing by prompt alone. OpenAI or Claude.

    python3 llm.py ask "my card hasn't arrived" [--open]
    python3 llm.py probe                  # which request options each candidate model accepts
    python3 llm.py pick [N] ['[{...}, ...]']   # compare setups on training messages, never test ones

The model gets its best setup: the 77 intents as a list with a one-line
description each (data/intents.json), structured outputs with the intents as an
enum so every answer is a valid label, and prompt caching. The intent list comes
first in the prompt and never changes, so it is the part that gets cached.

Two providers, picked by `provider` in llm-config.json, or else by whichever key
exists (OpenAI first):

- **OpenAI** (stdlib HTTP): temperature 0, `prompt_cache_key`, and a
  confidence from `logprobs`: the product of the probabilities of the tokens
  that spell the label. OpenAI caches a prefix of 1,024 tokens or more by itself.
- **Claude** (the official `anthropic` SDK): `output_config.format` for the
  schema, `cache_control` on the system prompt (Claude Haiku 5.5 caches from 512
  tokens), and an explicit `thinking` and `effort`. The Claude API returns no
  token probabilities, so the LLM has no confidence there; the router never
  needed one, because it reads ModernBERT's.

Keys: OPENAI_API_KEY or ANTHROPIC_API_KEY in .env.local (gitignored), and nowhere
else. Never printed. The Claude client is always pointed at api.anthropic.com, whatever ANTHROPIC_BASE_URL
says, so a tool that sets that variable for itself cannot reroute these calls.
"""

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
PROMPT_VERSION = "v1"

# USD per 1M tokens, standard tier, short prompts, read on 2026-11-23 from
# https://developers.openai.com/api/docs/pricing and
# https://platform.claude.com/docs/en/about-claude/pricing (Claude: prompts up
# to 100K tokens; ours are about 2K). OpenAI charges nothing extra to write its
# cache, so its cache_write is the input price.
PRICES_AS_OF = "2026-11-23"
PRICES = {
    "gpt-6-luna":       {"provider": "openai", "input": 0.10, "cache_write": 0.10, "cache_read": 0.01,  "output": 0.50},
    "gpt-5.6-luna":     {"provider": "openai", "input": 0.20, "cache_write": 0.20, "cache_read": 0.02,  "output": 1.20},
    "gpt-5.4-nano":     {"provider": "openai", "input": 0.20, "cache_write": 0.20, "cache_read": 0.02,  "output": 1.25},
    "gpt-5.4-mini":     {"provider": "openai", "input": 0.75, "cache_write": 0.75, "cache_read": 0.075, "output": 4.50},
    "gpt-5-nano":       {"provider": "openai", "input": 0.05, "cache_write": 0.05, "cache_read": 0.005, "output": 0.40},
    "gpt-5-mini":       {"provider": "openai", "input": 0.25, "cache_write": 0.25, "cache_read": 0.025, "output": 2.00},
    "gpt-4.1-mini":     {"provider": "openai", "input": 0.40, "cache_write": 0.40, "cache_read": 0.10,  "output": 1.60},
    "gpt-4o-mini":      {"provider": "openai", "input": 0.15, "cache_write": 0.15, "cache_read": 0.075, "output": 0.60},
    "claude-haiku-5-5": {"provider": "anthropic", "input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
}
PRICES_URL = {"openai": "https://developers.openai.com/api/docs/pricing",
              "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing"}

# The configuration bench.py and serve.py use, chosen by `pick` on training
# messages and saved in llm-config.json.
CONFIG_FILE = os.path.join(common.HERE, "llm-config.json")
DEFAULTS = {
    "openai": {"provider": "openai", "model": "gpt-6-luna", "temperature": 0, "reasoning_effort": None,
               "logprobs": True, "descriptions": True},
    # A classifier needs no deliberation; `pick` checks that against thinking on.
    "anthropic": {"provider": "anthropic", "model": "claude-haiku-5-5", "thinking": "disabled", "effort": "low",
                  "descriptions": True},
}

NONE = "none_of_these"
KEY_ENVS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def has_key(provider):
    return bool(common.env_local(KEY_ENVS[provider]))


def config():
    saved = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as fh:
            saved = json.load(fh)
    provider = saved.get("provider") or ("openai" if has_key("openai") or not has_key("anthropic") else "anthropic")
    return dict(DEFAULTS[provider], **saved)


# ── The prompt, the same for both providers ──────────────────────────────────

def system_prompt(descriptions=True, open_set=False):
    lines = ["You route customer support messages for a digital bank with a card, an app, "
             "top-ups, transfers and currency exchange.",
             "Read the customer's message and choose the one intent below that best matches it.",
             ""]
    if open_set:
        lines[1] = ("Read the customer's message and choose the one intent below that best matches it. "
                    f"If none of them fits, choose {NONE} and write a short, helpful reply to the customer "
                    "in `reply`. Otherwise leave `reply` empty.")
    lines.append("Intents:")
    for row in common.intents():
        lines.append(f"- {row['tool']}: {row['description']}" if descriptions else f"- {row['tool']}")
    if open_set:
        lines.append(f"- {NONE}: The message is not about any intent above.")
    lines += ["", "Answer with the intent's name exactly as written above."]
    return "\n".join(lines)


def schema(open_set=False):
    names = [row["tool"] for row in common.intents()] + ([NONE] if open_set else [])
    props = {"intent": {"type": "string", "enum": names}}
    if open_set:
        props["reply"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


TOOL_TO_LABEL = None


def to_label(tool):
    global TOOL_TO_LABEL
    if TOOL_TO_LABEL is None:
        TOOL_TO_LABEL = {row["tool"]: row["label"] for row in common.intents()}
    return NONE if tool == NONE else TOOL_TO_LABEL.get(tool, tool)


def cost(model, tokens):
    """USD for one call, from the normalised token counts every client returns."""
    p = PRICES[model]
    return (tokens["input"] * p["input"] + tokens["cache_write"] * p["cache_write"]
            + tokens["cache_read"] * p["cache_read"] + tokens["output"] * p["output"]) / 1e6


def label_confidence(content, logprobs):
    """exp(sum of logprobs) over the tokens that spell the intent value in the JSON answer."""
    if not logprobs:
        return None
    tokens = logprobs.get("content") or []
    try:
        value = json.loads(content)["intent"]
    except (ValueError, KeyError, TypeError):
        return None
    start = content.find(json.dumps(value))
    if start < 0:
        return None
    start, end = start + 1, start + 1 + len(value)
    pos, total, used = 0, 0.0, False
    for tok in tokens:
        text = tok.get("token", "")
        a, b = pos, pos + len(text)
        if b > start and a < end:
            total += tok.get("logprob", 0.0)
            used = True
        pos = b
    return math.exp(total) if used else None


def _result(cfg, parsed, ms, tokens, usage, served, confidence, extra, open_set):
    out = {"label": to_label(parsed["intent"]), "confidence": confidence, "ms": ms, "tokens": tokens,
           "usage": usage, "cost_usd": cost(cfg["model"], tokens), "model": served,
           "provider": cfg["provider"], **extra}
    if open_set:
        out["reply"] = parsed.get("reply", "")
    return out


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIClient:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        key = common.secret("OpenAI key", KEY_ENVS["openai"])
        self.http = common.Keepalive(OPENAI_BASE_URL, {"Authorization": f"Bearer {key}",
                                                       "User-Agent": "decision-models/1.0"})
        self._prompts = {}

    def body(self, text, open_set=False):
        cfg = self.cfg
        key = (cfg["descriptions"], open_set)
        if key not in self._prompts:
            self._prompts[key] = system_prompt(cfg["descriptions"], open_set)
        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": self._prompts[key]},
                         {"role": "user", "content": text}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "route_open" if open_set else "route", "strict": True, "schema": schema(open_set)}},
            "prompt_cache_key": f"decision-models-banking77-{PROMPT_VERSION}-{'open' if open_set else 'closed'}",
        }
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if cfg.get("reasoning_effort"):
            body["reasoning_effort"] = cfg["reasoning_effort"]
        if cfg.get("logprobs"):
            body["logprobs"] = True
        return body

    def ask(self, text, open_set=False):
        status, res, headers, ms = self.http.post("/chat/completions", self.body(text, open_set))
        if status != 200:
            detail = res.get("error", {}).get("message") if isinstance(res, dict) else str(res)[:300]
            raise RuntimeError(f"OpenAI {status}: {detail}")
        choice = res["choices"][0]
        content = choice["message"].get("content") or ""
        usage = res.get("usage", {})
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        tokens = {"input": usage.get("prompt_tokens", 0) - cached, "cache_write": 0, "cache_read": cached,
                  "output": usage.get("completion_tokens", 0),
                  "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0}
        return _result(self.cfg, json.loads(content), ms, tokens, usage, res.get("model"),
                       label_confidence(content, choice.get("logprobs")),
                       {"raw": {"content": content}}, open_set)


# ── Claude ───────────────────────────────────────────────────────────────────

class AnthropicClient:
    def __init__(self, cfg):
        import anthropic
        self.cfg = dict(cfg)
        key = common.secret("Claude API key", KEY_ENVS["anthropic"])
        # One client, one connection pool: calls after the first reuse the connection.
        # No SDK retries, so a timed call is one request; bench.py retries failures itself.
        self.client = anthropic.Anthropic(api_key=key, base_url=ANTHROPIC_BASE_URL, max_retries=0, timeout=30.0)
        self._prompts = {}

    def request(self, text, open_set=False):
        cfg = self.cfg
        key = (cfg["descriptions"], open_set)
        if key not in self._prompts:
            self._prompts[key] = system_prompt(cfg["descriptions"], open_set)
        thinking = cfg.get("thinking", "disabled")
        req = {
            "model": cfg["model"],
            # Room for a reply on the open prompt, and for thinking when it is on.
            # 8,000 for thinking: in setup trials, effort max ran past 4,000 on 4 of 300 messages.
            "max_tokens": (400 if open_set else 64) + (0 if thinking == "disabled" else 8000),
            "system": [{"type": "text", "text": self._prompts[key], "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": text}],
            "output_config": {"format": {"type": "json_schema", "schema": schema(open_set)},
                              "effort": cfg.get("effort", "low")},
            "thinking": {"type": thinking},
        }
        return req

    def ask(self, text, open_set=False):
        import anthropic
        start = time.perf_counter()
        try:
            res = self.client.messages.create(**self.request(text, open_set))
        except anthropic.APIStatusError as err:
            raise RuntimeError(f"Claude {err.status_code}: {str(err.message)[:300]}") from err
        except anthropic.APIConnectionError as err:
            raise RuntimeError(f"Claude unreachable: {str(err)[:200]}") from err
        ms = (time.perf_counter() - start) * 1000
        if res.stop_reason not in ("end_turn", "stop_sequence"):
            raise RuntimeError(f"Claude stopped with {res.stop_reason}")
        content = next(b.text for b in res.content if b.type == "text")
        u = res.usage
        tokens = {"input": u.input_tokens or 0, "cache_write": u.cache_creation_input_tokens or 0,
                  "cache_read": u.cache_read_input_tokens or 0, "output": u.output_tokens or 0, "reasoning": None}
        return _result(self.cfg, json.loads(content), ms, tokens, u.to_dict(), res.model, None,
                       {"raw": {"content": content, "request_id": getattr(res, "_request_id", None)}}, open_set)


def client(cfg=None):
    cfg = dict(cfg or config())
    cfg = dict(DEFAULTS[cfg.get("provider", "openai")], **cfg)
    return AnthropicClient(cfg) if cfg["provider"] == "anthropic" else OpenAIClient(cfg)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ask(args):
    open_set = "--open" in args
    text = " ".join(a for a in args if a != "--open")
    print(json.dumps(client().ask(text, open_set), indent=2, default=str))


PROBES = {
    "openai": [{"temperature": 0, "logprobs": True}, {"temperature": None, "logprobs": True},
               {"temperature": None, "logprobs": False},
               {"temperature": None, "logprobs": False, "reasoning_effort": "minimal"},
               {"temperature": None, "logprobs": False, "reasoning_effort": "none"}],
    "anthropic": [{"thinking": "disabled", "effort": "low"}, {"thinking": "adaptive", "effort": "low"}],
}


def cmd_probe(_args):
    """Which options each candidate the available key can reach accepts, on one training message."""
    text = common.split("train")[0]["text"]
    for model, price in PRICES.items():
        provider = price["provider"]
        if not has_key(provider):
            continue
        for variant in PROBES[provider]:
            cfg = dict(DEFAULTS[provider], model=model, **variant)
            try:
                r = client(cfg).ask(text)
                t = r["tokens"]
                print(f"OK   {model:16} {json.dumps(variant):70} {r['label']:22} {r['ms']:6.0f}ms "
                      f"conf={r['confidence']} in/write/read/out={t['input']}/{t['cache_write']}/"
                      f"{t['cache_read']}/{t['output']} served={r['model']}")
                break
            except Exception as err:  # noqa: BLE001
                print(f"no   {model:16} {json.dumps(variant):70} {str(err)[:120]}")


def cmd_pick(args):
    """Accuracy, latency and cost of each setup on N training messages (default 300).

    Training messages, never test ones, so choosing the LLM's setup never sees the
    messages it will be scored on.
    """
    n = int(args[0]) if args else 300
    cands = json.loads(args[1]) if len(args) > 1 else [config()]
    rows = random.Random(77).sample(common.split("train"), n)
    for cfg in cands:
        c = client(cfg)
        right, ms, usd, errors = 0, [], 0.0, 0
        for r in rows:
            try:
                out = c.ask(r["text"])
            except Exception as err:  # noqa: BLE001
                errors += 1
                print("  error", str(err)[:200])
                continue
            right += out["label"] == r["label"]
            ms.append(out["ms"])
            usd += out["cost_usd"]
        print(json.dumps({"config": c.cfg, "n": n, "right": right, "errors": errors,
                          "p50_ms": round(common.percentile(ms, 50)) if ms else None,
                          "p95_ms": round(common.percentile(ms, 95)) if ms else None,
                          "usd_per_10k": round(usd / max(1, len(ms)) * 10000, 4)}), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"ask": cmd_ask, "probe": cmd_probe, "pick": cmd_pick}.get(cmd, lambda _a: sys.exit(__doc__))(sys.argv[2:])
