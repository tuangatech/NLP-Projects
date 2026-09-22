# Intent Detection: RoBERTa vs. Jev

Two approaches to the same problem — classifying customer support utterances into one of 27 intents — compared side by side: a fine-tuned transformer and a zero/few-shot call to a hosted evaluation model.

## Dataset

[Customer Support Intent Dataset](https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset/data) (Kaggle), split into:

| File | Rows | Purpose |
|---|---|---|
| `dataset/data_train.csv` | 6,539 | Training (RoBERTa) / few-shot examples (Jev) |
| `dataset/data_test.csv` | 818 | Held-out evaluation |

(The original split also included `data_validation.csv`, used only for RoBERTa's early stopping — dropped since it serves no purpose for Jev's training-free approach.)

Each row: `utterance, intent, category, tags`. 27 intents (e.g. `cancel_order`, `get_refund`, `track_order`), fairly balanced (~225–270 rows each in training).

## Approach 1: Fine-tuned RoBERTa (prior experiment, notebook not included in this repo)

Trains a `RobertaForSequenceClassification` head on top of `roberta-base`:

- Preprocessing: lowercase, stopword removal (no lemmatizing/stemming — didn't help).
- Config: `max_len=64`, `batch_size=16`, `lr=2e-5`, 5 epochs, dropout 0.3, early stopping (patience 3).
- Training loop: AdamW + cross-entropy loss + cosine annealing LR schedule + gradient clipping.
- Also explores text augmentation (synonym replacement via GloVe, back-translation, pattern-based paraphrasing) as a way to grow/diversify the training set, though augmented data isn't folded back into the training run in the notebook.

**Result:** 99.76% validation accuracy (best epoch, 5/5). The notebook does not compute an aggregate accuracy on `data_test.csv` — it only spot-checks 10 random test rows (all correct), so there's no formal test-set number to cite for this approach. This asymmetry is worth keeping in mind when comparing against Jev's test-set accuracy below: the RoBERTa model was also directly trained on this data, so it isn't a fair apples-to-apples baseline — it's a full-supervision upper bound to keep in view.

## Approach 2: Jev (few-shot, no training)

[`classify.mts`](classify.mts)

[Jev](https://vercel.com/i/jev-integrations) (`typesafe-ai/jev`) is an evaluation model available through [Vercel AI Gateway](https://vercel.com/docs/ai-gateway/getting-started) — it doesn't generate free text, it answers typed questions (`boolean` / `choice` / `score`) against a piece of state. Classification maps directly onto its `choice` type, so there's no training step at all: every test utterance is classified live via one API call.

### How it works

1. **State** — the customer utterance being classified.
2. **Question** — a single `choice` question named `intent`, whose `criteria` is a map of `{ intent_name: description }` for all 27 intents.
3. **Few-shot grounding** — since the `choice` type has no dedicated examples field, each intent's description is built from 3 real training utterances for that intent, e.g.:
   ```
   cancel_order: e.g., "would it be possible to cancel the order I made?"; "problem with cancelling orders"; "will you give me information about canceling an order?"
   ```
4. Jev returns the winning `choice` plus a probability distribution over all 27 options.

### Diverse example selection

The first version picked the first 3 training rows per intent verbatim — which, on inspection, were frequently near-duplicates ("cancelling order" / "cancel order" / "canceling the order"), giving the model weak, redundant grounding. This was replaced with **greedy farthest-point selection**: starting from the first example, each next example is the one with the lowest word-overlap (Jaccard similarity) to everything already picked, so the 3 examples per intent span different phrasings (question / complaint / info-request) instead of restating the same sentence. See `pickDiverseExamples()` in `classify.mts`.

### Evaluation setup

- **Sample:** stratified — the first 6 rows per intent from `data_test.csv` (~162 rows across 27 intents), so every intent gets evaluated, not just the majority ones.
- **Concurrency:** 3 requests in flight, staggered with a 300ms minimum interval between dispatches (`throttle()`), to stay under the gateway's rate limits.
- **Timeouts & retries:** each call gets a 20s timeout (`AbortController`) so one stuck request can't hang the whole batch; failures retry up to 2 more times with backoff (longer for `GatewayRateLimitError` specifically).
- **Output:** live progress (`[N/162] predicted_intent`), a console summary (overall + per-intent accuracy), and `results/jev_predictions.csv` with one row per sample plus an accuracy summary appended as `#`-prefixed comment lines.

**Result:** 98.1% accuracy (160/162 scored, 2 network errors) on the stratified test sample — competitive with RoBERTa's validation number, achieved with zero training and a handful of examples per intent.

## Setup

Prerequisites: Node.js 22.18+ (uses native TypeScript execution).

1. Add a Vercel AI Gateway key to `.env.local` (gitignored):
   ```
   AI_GATEWAY_API_KEY=your_key_here
   ```
   Create one at [vercel.com/.../ai-gateway/api-keys](https://vercel.com/docs/ai-gateway/getting-started#create-an-api-key). Your Vercel team needs a payment method on file to unlock AI Gateway credits.
2. Install dependencies:
   ```
   npm install
   ```
3. Run the classifier:
   ```
   npm run classify
   ```

## Files

```
dataset/                        Kaggle customer support intent dataset (train/test)
classify.mts                    Few-shot intent classification via Jev / AI Gateway
results/jev_predictions.csv     Per-row Jev predictions + accuracy summary (generated)
```

## Takeaways

- Jev's `choice` evaluation type is a near-perfect fit for intent classification: no separate few-shot API, but a hand-built `criteria` description per label does the job.
- Example *diversity* mattered more than example *count* — de-duplicating near-identical few-shot examples per intent improved grounding without adding more of them.
- The AI Gateway backend's per-request latency is inconsistent (hundreds of ms to 15s+ observed) and occasionally rate-limits under concurrency — a client-side timeout, retry-with-backoff, and request throttle are necessary for a reliable batch run, not optional polish.
- RoBERTa's reported number (99.76%) is validation accuracy from training, not a held-out test accuracy, and the model was directly fine-tuned on this data — so it's a ceiling reference more than a fair baseline, not a metric on equal footing with Jev's zero-shot test result.
