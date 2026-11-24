# Decision models demo

The demo from the talk **"Decision models for developers who use LLMs"**: one
customer support message, 77 possible intents, and three ways to pick one.

| Model | What it is given | Where it runs |
|---|---|---|
| **LLM**: a small hosted model, prompt only (Claude Haiku 5.5, or one of OpenAI's small models) | The 77 intents, one line each, as an enum it must choose from. No examples | Anthropic's (or OpenAI's) API |
| **ModernBERT**: ModernBERT-base (149M), fine-tuned as a classifier | 10,003 labelled messages. Returns a probability for every intent | Your machine (measured on an Apple M4 laptop) |
| **Fern**: Fern Pinna (1.5B), fine-tuned | The same 10,003 messages, as tool calls: one tool per intent | The Fern API, on one GPU in UK South |

On all 3,080 test messages, run one at a time on 23 November 2026:

| | LLM | ModernBERT | Fern |
|---|---|---|---|
| Accuracy | 86.0% | **93.6%** | 90.5% |
| p50 latency | 1,109 ms | **13 ms** | 261 ms |
| Cost per 10,000 calls | $0.8263 | **$0.0461** | N/A (flat $49 a month) |
| Training examples | 0 | 10,003 | 10,003 |

Fern has no price per call: Fernfly charges a flat monthly fee, whatever the
volume. ModernBERT's figure prices its measured seconds at a cloud machine with
the same chip, kept busy.

Letting ModernBERT answer when it is more than 0.9 sure, and sending the rest to
the LLM: ModernBERT handles 90.6% of messages, and the blend scores **93.9% at
$0.1516 per 10,000 calls**.

[`RESULTS.md`](RESULTS.md) has the full report, and
[`results.json`](results.json) the same in JSON: model ids, run times, the
pricing behind every cost, what was tried and dropped, and how each number was
measured. Nothing is rounded in a model's favour: accuracy is rounded down,
latency and cost up.

Data: [Banking77](https://huggingface.co/datasets/PolyAI/banking77) by PolyAI
(Casanueva et al., 2020), CC BY 4.0: 10,003 training and 3,080 test messages,
included under [`data/`](data/).

## Run it

Python 3.12 or later. ModernBERT trains and runs on Apple silicon (MPS), or on
the CPU, slowly. The LLM, Fern and the page's service need only the standard
library.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Train ModernBERT** (19 minutes on an M4; the checkpoint goes to
`models/bert/`, which git ignores):

```bash
.venv/bin/python bert.py train
```

**Start the page**, then open <http://127.0.0.1:8765/>:

```bash
.venv/bin/python serve.py
```

That much needs no keys: ModernBERT answers everything, and the results view
shows the run above.

### The LLM

Put a key in `.env.local` in this folder, which git ignores:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`OPENAI_API_KEY` works instead; with both present, `provider` in
`llm-config.json` decides. Keys are read from `.env.local` and nowhere else, and
nothing here prints one. Use a key from a separate project with a spend limit: a
full benchmark run costs under a dollar.

The setup is in `llm-config.json`: Claude Haiku 5.5 with thinking on, effort
high, and the intent descriptions. `llm-pick.json` has the seven setups it was
chosen from, all scored on the same 300 training messages. `llm.py probe`
shows which request options a model accepts; `llm.py pick` scores setups on
training messages, never test ones. Prices live in `PRICES` in `llm.py`: update
them, with the date, from the provider's pricing page. Claude returns no token
probabilities, so the LLM shows no confidence there; the router is unaffected,
because it reads ModernBERT's.

### Fern

The Fern model in the results is Fern Pinna (1.5B), fine-tuned on the 10,003
training messages with Fernfly's default settings. To make the same model in
[Fernfly](https://fernfly.com/):

1. Export the two files it needs, into `export/`:

   ```bash
   .venv/bin/python fern.py export
   ```

2. In Fernfly, create a project with **OpenAPI / Swagger** as the source and
   upload `export/banking77.openapi.json`: one tool per intent, 77 in all, with
   no parameters.
3. Keep all 77 tools selected.
4. At the generate step, choose **Import JSONL** and upload
   `export/banking77.pairs.jsonl` instead of generating: the 10,003 training
   messages, each paired with its intent's tool.
5. At the train step, choose **Fern Pinna** and keep the default settings.
6. Train, then deploy.
7. On the project's page, turn **AI fallback** off, so a message the model
   cannot route shows as a miss rather than being answered by another model.

### Run the benchmark

About 90 minutes, almost all of it the LLM: 3,080 calls at about 1.3 seconds
each, then about 290 more for the router. It writes `RESULTS.md`,
`results.json` and the page's replay cache:

```bash
.venv/bin/python bench.py run
```

If the network drops part-way, resume the same run rather than start again:
`bench.py run --resume runs/<timestamp>`. `--only llm`, `--only modernbert`
or `--only fern` runs one model; `--n 50` is a smoke test that writes no
report.

## The page

Six views; the tabs or the keys `1`–`6` switch between them, and the arrow
keys step through.

1. **The task.** One message, the 77 intents as dots (hover one for its name),
   and what each model was given.
2. **Live.** `Space` sends 20 test messages to all three models, one at a time,
   side by side: the answer, right or wrong, confidence and time. `S` draws a
   new 20.
3. **Full test set.** The benchmark's results, and the routing row.
4. **Not one of the 77.** Type a banking request no intent covers, then
   `Enter`; `Tab` fills in examples. ModernBERT's confidence decides whether the
   router sends it to the LLM, which can answer "none of these" and reply.
5. **Fix a mistake.** A test message ModernBERT gets wrong. `Enter` asks it;
   `R` adds the corrected example, retrains for ten steps, asks again and
   reruns all 3,080 test messages, in about 25 seconds; `X` puts the original
   model back.
6. **Try it.** This repository, and the credits.

`L` (or the button top right) switches light and dark. `C` switches the LLM and
Fern to answers recorded earlier, from the benchmark run and from the page's own
use, so the page works without a network; they are marked **cached**.
`serve.py --cached` starts that way, and `serve.py --no-bert` starts without
torch (then view 5 does not work).

## What to edit

| To change | Edit |
|---|---|
| The message on view 1, the examples on view 4, the message on view 5, the repository link | `demo.json` |
| The 20 messages on view 2 | `live_seed` in `demo.json` (a seeded random sample of the test set, not hand-picked) |
| The intent descriptions the LLM and Fern see | `data/intents.json` (then retrain Fern and rerun the benchmark) |
| The LLM's model or prompt | `llm-config.json`, `system_prompt()` in `llm.py` |
| ModernBERT's training recipe | `RECIPE` in `bert.py`; view 5's retrain in `RETRAIN` |
| The prices | `PRICES` in `llm.py`, `INSTANCE` and `PLAN` in `bench.py` |
| The page | `web/index.html`, `web/app.css`, `web/app.js` |

## Files

| File | |
|---|---|
| `common.py` | Data, labels, credentials, one kept-alive HTTPS connection |
| `prepare_data.py` | Turns the Banking77 parquet files into `data/*.jsonl` |
| `bert.py` | ModernBERT: train, evaluate, ask; and `retrain_one`, view 5's retrain |
| `llm.py` | The LLM: prompt, structured outputs, prompt caching, confidence from logprobs (OpenAI), cost from usage |
| `fern.py` | Fern: the export for Fernfly, and asking the model |
| `bench.py` | The full run, and `RESULTS.md` |
| `rehearse_retrain.py` | Finds a test message one retrain reliably fixes, and a recipe that does not cost accuracy |
| `out_of_set.py` | Twenty requests that fit none of the 77: does ModernBERT know it does not know? |
| `llm-config.json`, `llm-pick.json` | The LLM's setup, and the setups it was chosen from |
| `demo.json` | The messages each view uses |
| `serve.py` | The page's service: ModernBERT in-process, the LLM and Fern over kept-alive connections, the cache, the retrain |
| `web/` | The page: plain HTML, CSS and JavaScript, no build step, nothing fetched from the internet |

## How the measurements are made

- **Accuracy**: right answers over all 3,080 test messages. A call that fails
  three times is scored wrong.
- **Latency**: wall time per call, one message at a time, after warm-up, over a
  kept-alive connection (so no call pays for a TLS handshake). p50 and p95 are
  nearest-rank. The LLM and Fern include the network; Fern's includes a round
  trip to the UK. ModernBERT's is in-process, because it runs where the app runs.
- **Cost per 10,000 calls**: for the LLM, the API's own token counts (cached
  tokens at the cached price, reasoning tokens as output) times the list price
  on the day. For ModernBERT, mean seconds per call times 10,000 times the hourly
  price of a cloud machine with the same chip (`mac-m4.metal`), divided by
  3,600: one call at a time, on a machine kept busy. Fern has none: Fernfly
  charges a flat $49 a month (Pro), whatever the volume.
- **Nothing is tuned on the test set.** ModernBERT's recipe is fixed, with no
  early stopping. The LLM's setup was chosen on training messages. Fern uses
  Fernfly's default settings for this many examples. View 5's message and
  retrain recipe were chosen by trying test messages, which is fine for showing
  a retrain and never touches the benchmark.
- **Fern can answer with an intent that does not exist**, because it generates
  the tool name. RESULTS.md counts those; they are scored wrong.
- The split is Banking77's own. Six test messages also appear, word for word,
  in the training set; they are left in, as every published result does.

## Licence and credits

The code is MIT-licensed: see [`LICENSE`](LICENSE). The data under `data/` is
Banking77 by PolyAI (Casanueva et al., 2020), under its own licence, CC BY 4.0.
ModernBERT is by Answer.AI and LightOn, Apache 2.0. Fern and Fernfly are by
[antelligent](https://antelligent.org/).
