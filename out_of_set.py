#!/usr/bin/env python3
"""
out_of_set.py: how often ModernBERT knows it does not know.

    python3 out_of_set.py            # ModernBERT on every message below; the LLM's open prompt too, with a key

Twenty banking requests we wrote that fit none of Banking77's 77 intents. For
each, ModernBERT's top intent and confidence, and whether the router (ModernBERT
above 0.9, else the LLM) would have sent it on. With an LLM key, the LLM's
answer to the ones that go to it as well. Written to out-of-set.json, which
bench.py's report includes in RESULTS.md. ModernBERT is a classifier with no
"none of these" label, so a confident wrong answer here is the failure that
matters.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

THRESHOLD = 0.9
OUT = os.path.join(common.HERE, "out-of-set.json")

MESSAGES = [
    "Can I get a home loan to buy a flat in Bengaluru?",
    "What interest rate do you pay on savings?",
    "Do you offer business accounts for my startup?",
    "Can I add my wife as a joint account holder?",
    "Do you have an overdraft facility?",
    "Can I buy travel insurance through the app?",
    "How do I invest in stocks with my account?",
    "Can you send me a bank statement for my visa application?",
    "Is my money protected if the bank goes bust?",
    "Can I pay my credit card bill with this account?",
    "What are your customer service hours?",
    "Can I get a cheque book?",
    "Do you offer student loans?",
    "Can I set a savings goal for my holiday?",
    "What's the IFSC code for my account?",
    "Can I split a restaurant bill with friends in the app?",
    "I'd like to speak to a human, please.",
    "Do you have a branch in Koramangala?",
    "Can I get a loan against my salary?",
    "How do I open a fixed deposit?",
]


def main():
    import bert
    model, tok = bert.load()
    llm_client = None
    try:
        import llm
        llm_client = llm.client()
    except BaseException as err:  # noqa: BLE001  (no key: ModernBERT only)
        print(f"LLM not asked: {str(err)[:100]}")
    rows = []
    for text in MESSAGES:
        guess = bert.ask_one(model, tok, text)
        row = {"text": text, "modernbert_label": guess["label"], "modernbert_confidence": guess["confidence"],
               "modernbert_top": guess["top"], "routed": guess["confidence"] <= THRESHOLD}
        if llm_client and row["routed"]:
            answer = llm_client.ask(text, open_set=True)
            row.update(llm_label=answer["label"], llm_reply=answer.get("reply"), llm_ms=answer["ms"])
        rows.append(row)
        print(f"{guess['confidence']:.2f}  {'to the LLM' if row['routed'] else 'ModernBERT':10}  {text:60}  "
              f"{guess['label']}" + (f"  | LLM: {row.get('llm_label')}" if row.get("llm_label") else ""))
    out = {"threshold": THRESHOLD, "messages": rows, "routed": sum(r["routed"] for r in rows),
           "modernbert_confident_and_wrong": [r for r in rows if not r["routed"]]}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"{out['routed']}/{len(rows)} went to the LLM -> {OUT}")


if __name__ == "__main__":
    main()
