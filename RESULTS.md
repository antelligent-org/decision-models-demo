# Results: three ways to route a support message

Run **2026-11-23T1418Z**, started 2026-11-23T14:18:43Z (2026-11-23 19:48 IST). All 3,080 test messages of Banking77 by PolyAI (Casanueva et al., 2020), CC BY 4.0, 77 intents, sent one at a time. Written by `bench.py report`; do not edit by hand.

| | LLM, prompt only | ModernBERT-base, fine-tuned | Fern, fine-tuned |
|---|---:|---:|---:|
| Model | claude-haiku-5-5 | ModernBERT-base (149M) | Fern Pinna (`fern-1500`) |
| Accuracy | **86.0%** (2,649/3,080) | **93.6%** (2,883/3,080) | **90.5%** (2,789/3,080) |
| p50 latency | 1,109 ms | 13 ms | 261 ms |
| p95 latency | 2,126 ms | 18 ms | 304 ms |
| Cost per 10,000 calls | $0.8263 | $0.0461 | N/A (flat $49 a month, Fernfly Pro) |
| Training examples | 0 | 10,003 | 10,003 |
| Answers that are not one of the 77 | 0 | 0 | 11 |
| Errors (scored wrong) | 0 | 0 | 0 |

Each model ran on its own, one message at a time: LLM 2026-11-23T17:10:53Z to 2026-11-23T18:14:09Z; ModernBERT 2026-11-23T14:18:50Z to 2026-11-23T14:19:31Z; Fern 2026-11-23T14:19:32Z to 2026-11-23T14:33:01Z (UTC).

Nothing is rounded in a model's favour: accuracy is rounded down, latency and cost are rounded up. The exact counts are beside each accuracy, and every raw answer is in `results.json` and the run directory.

Fern has no price per call to set beside the others: Fernfly charges a flat monthly fee, whatever the volume.

## Routing: ModernBERT answers when it is sure, the LLM takes the rest

ModernBERT answers when its confidence is above 0.9; everything else goes to the LLM.

| Share ModernBERT handles | ModernBERT accuracy on its share | LLM accuracy on the rest | Blended accuracy | Blended cost per 10,000 |
|---:|---:|---:|---:|---:|
| 90.6% (2,792) | 97.7% | 57.6% (288 messages) | **93.9%** (2,894/3,080) | $0.1516 |

Blended cost is ModernBERT on every message (it always runs first) plus the LLM on the share passed to it.

The page's router sends the rest to the LLM with one more option, `none_of_these`, and a free-text reply, which is what lets it handle a message no intent covers. With that prompt the LLM gets 156/288 of the rest right (every test message has an intent, so `none_of_these` counts as wrong), for a blended 93.6% at $0.1699 per 10,000.

## Requests that fit none of the 77

20 banking requests we wrote (`out_of_set.py`). ModernBERT has no "none of these" label, so the router's only defence is its confidence. 18 of 20 fell to 0.9 or below and went to the LLM. 2 did not: ModernBERT was sure, and wrong.

- "Can I get a cheque book?": ModernBERT said `top_up_by_cash_or_cheque` at 0.98.
- "Can I get a loan against my salary?": ModernBERT said `receiving_money` at 1.00.

With the router's open prompt (the LLM's, plus `none_of_these` and a reply), the LLM answered `none_of_these` on 18 of the 18 it was sent, and wrote a reply to each. Some replies state bank policy the prompt never gave it ("we don't offer joint accounts"): the open-ended path is the LLM's to handle, but in a product it needs the bank's real policies to answer from.

## Which LLM setup

The LLM's setup was chosen on 300 training messages (never test ones), the same 300 for each candidate. The chosen one is in bold.

| Setup | Right | p50 | p95 | Cost per 10,000 |
|---|---:|---:|---:|---:|
| `claude-haiku-5-5`, thinking off, effort low, intent descriptions | 243/300 | 1,017 ms | 1,588 ms | $0.4482 |
| `claude-haiku-5-5`, thinking off, effort low, intent names only | 225/300 | 969 ms | 1,667 ms | $0.3244 |
| `claude-haiku-5-5`, thinking on, effort low, intent descriptions | 246/300 | 970 ms | 1,894 ms | $0.5821 |
| `claude-haiku-5-5`, thinking on, effort medium, intent descriptions | 249/300 | 1,052 ms | 1,930 ms | $0.7297 |
| **`claude-haiku-5-5`, thinking on, effort high, intent descriptions** | 253/300 | 1,207 ms | 2,111 ms | $0.8546 |
| `claude-haiku-5-5`, thinking on, effort xhigh, intent descriptions | 252/300 | 1,208 ms | 2,284 ms | $1.02 |
| `claude-haiku-5-5`, thinking on, effort max, intent descriptions (4 errors) | 253/300 | 1,612 ms | 6,839 ms | $2.41 |

Chosen for accuracy: the best measured, tied with effort max, which was slower, dearer and failed 4 calls. The differences between the thinking-on setups are within sampling noise on 300 messages; the intent descriptions are not (+18).

## How each number was measured

### LLM, prompt only

- Model `claude-haiku-5-5` (served as claude-haiku-5-5): thinking adaptive, effort high, structured outputs (`output_config.format`) with the 77 intents as an enum, the intent list cached with `cache_control`. The Claude API returns no token probabilities, so the LLM has no confidence. Prompt `v1` (names and one-line descriptions).
- Tokens per call, from the API's usage fields: 3,391 input (99% of them read from the cache), 93.3 output.
- List price on 2026-11-23, per 1M tokens: $0.10 input, $0.125 cache write, $0.01 cache read, $0.50 output (https://platform.claude.com/docs/en/about-claude/pricing).
- Cost per 10,000 calls: per call = (uncached input tokens x $0.10 + cache writes x $0.125 + cache reads x $0.01 + output x $0.50) / 1M, from the API's own usage fields; summed over the run, divided by calls, x 10,000 = **$0.8263**.
- p50 1,109 ms, p95 2,126 ms, mean 1,228 ms (nearest-rank percentiles), 0 errors, 0 not run.

### ModernBERT-base, fine-tuned

- `answerdotai/ModernBERT-base` at revision `8949b909ec90`, fine-tuned for 5 epochs (1565 steps) on 10,003 messages in 19 minutes on mps (torch 2.15.0). No early stopping, nothing tuned on the test set.
- Run on a laptop (mps), one message at a time, in-process: no network hop, because ModernBERT runs where the app runs.
- Cost per 10,000 calls: mean seconds per call (0.0135) x 10,000 x $1.23/hour (AWS EC2 mac-m4.metal) / 3,600 = **$0.0461**.
- AWS EC2 mac-m4.metal: Apple M4 (10 vCPU, 24 GiB), $1.23/hour on demand, us-east-1, https://instances.vantage.sh/aws/ec2/mac-m4.metal (2026-11-23). Same chip as the laptop ModernBERT was timed on. AWS bills Mac hosts per second after a 24-hour minimum, so this per-call figure assumes the machine is kept busy.
- p50 13 ms, p95 18 ms, mean 14 ms (nearest-rank percentiles), 0 errors, 0 not run.

### Fern, fine-tuned

- Fern Pinna (`fern-1500`), fine-tuned on 10,003 messages, hyperparameters {"n_epochs": 2, "batch_size": 16, "learning_rate": 0.0002} (Fernfly's defaults for this many pairs). One tool per intent, no arguments.
- Fern Pinna learns its tools in training, so at inference it is sent only the message.
- Served from one GPU in UK South, so every latency includes a round trip from the laptop to the UK and back.
- Fern is generative, so it can name a tool it was never given: it did on 11 of the 3,080 messages (scored wrong). The LLM cannot, because structured outputs hold it to the 77; ModernBERT cannot, because it is a classifier over the 77.
- Cost per 10,000 calls: N/A. Fernfly charges a flat $49 a month (Fernfly Pro, unlimited requests, subject to fair use), whatever the volume (https://fernfly.com/pricing, 2026-11-24): not by compute, tokens or calls.
- p50 261 ms, p95 304 ms, mean 263 ms (nearest-rank percentiles), 0 errors, 0 not run.

## Data

Banking77 by PolyAI (Casanueva et al., 2020), CC BY 4.0: https://huggingface.co/datasets/PolyAI/banking77 at revision `1846984189f2`. 10,003 training and 3,080 test messages.
