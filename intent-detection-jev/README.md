# Intent Detection: RoBERTa vs. Jev

Two approaches to the same problem — classifying customer support utterances into one of 27 intents — compared side by side: a fine-tuned transformer and a zero/few-shot call to a hosted evaluation model.

## Dataset

[Customer Support Intent Dataset](https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset/data) (Kaggle), split into:

| File | Rows | Purpose |
|---|---|---|
| `dataset/examples.csv` | 6,539 | Training data for RoBERTa (prior experiment) / few-shot example pool for Jev |
| `dataset/eval.csv` | 818 | Held-out evaluation |

(The original Kaggle split names these `data_train.csv` / `data_validation.csv` / `data_test.csv`. Renamed here to `examples.csv` / `eval.csv` since Jev does no training — `examples.csv` is really just the pool `pickDiverseExamples()` draws few-shot examples from. `data_validation.csv` is dropped entirely: it only served RoBERTa's early stopping and has no role in Jev's training-free approach.)

Each row: `utterance, intent, category, tags`. 27 intents (e.g. `cancel_order`, `get_refund`, `track_order`), fairly balanced (~225–270 rows each in training).

## Approach 1: Fine-tuned RoBERTa ([notebook](../Intent%20Detection%20w%20RoBERTa/Fine-Tuning%20RoBERTa%20for%20Intent%20Recognition.ipynb))

Trains a `RobertaForSequenceClassification` head on top of `roberta-base`:

- Preprocessing: lowercase, stopword removal (no lemmatizing/stemming — didn't help).
- Config: `max_len=64`, `batch_size=16`, `lr=2e-5`, 5 epochs, dropout 0.3, early stopping (patience 3).
- Training loop: AdamW + cross-entropy loss + cosine annealing LR schedule + gradient clipping.
- Also explores text augmentation (synonym replacement via GloVe, back-translation, pattern-based paraphrasing) as a way to grow/diversify the training set, though augmented data isn't folded back into the training run in the notebook.

**Result** (RTX 4070 Laptop GPU, 5 epochs, ~50 s/epoch):

- **Training + validation time:** 255.5 s (4.3 min) in total.
- **Validation accuracy:** 99.63% (best epoch; used only for early stopping and checkpoint selection).
- **Test accuracy, full held-out test set:** 99.88% (817/818).
- **Test accuracy, same 162-row stratified sample as Jev** (first 6 rows per intent): 100% (162/162).
- **Inference:** 1.4 s for all 818 test rows, ~1.7 ms/sample (batched, local GPU).

The test set was never used for training or checkpoint selection, so these are genuine held-out numbers. The model was still fully supervised on ~6.5k labeled examples of this exact distribution, so it remains an upper-bound reference for Jev's few-shot approach rather than a like-for-like baseline.

## Approach 2: Jev (few-shot, no training)

[`classify.mts`](classify.mts)

[Jev](https://vercel.com/i/jev-integrations) (`typesafe-ai/jev`) is an evaluation model available through [Vercel AI Gateway](https://vercel.com/docs/ai-gateway/getting-started) — it doesn't generate free text, it answers typed questions (`boolean` / `choice` / `score`) against a piece of state. Classification maps directly onto its `choice` type, so there's no training step at all: every test utterance is classified live via one API call.

### How it works

1. **State** — the customer utterance being classified.
2. **Question** — a single `choice` question named `intent`, whose `criteria` is a map of `{ intent_name: description }` for all 27 intents.
3. **Few-shot grounding** — since the `choice` type has no dedicated examples field, each intent's description is built from 3 real example-pool utterances for that intent, e.g.:
   ```
   cancel_order: e.g., "would it be possible to cancel the order I made?"; "problem with cancelling orders"; "will you give me information about canceling an order?"
   ```
4. Jev returns the winning `choice` plus a probability distribution over all 27 options.

### Diverse example selection

The first version picked the first 3 example-pool rows per intent verbatim — which, on inspection, were frequently near-duplicates ("cancelling order" / "cancel order" / "canceling the order"), giving the model weak, redundant grounding. This was replaced with **greedy farthest-point selection**: starting from the first example, each next example is the one with the lowest word-overlap (Jaccard similarity) to everything already picked, so the 3 examples per intent span different phrasings (question / complaint / info-request) instead of restating the same sentence. See `pickDiverseExamples()` in `classify.mts`.

### Evaluation setup

- **Sample:** stratified — the first 6 rows per intent from `eval.csv` (~162 rows across 27 intents), so every intent gets evaluated, not just the majority ones.
- **Concurrency:** 3 requests in flight, staggered with a 300ms minimum interval between dispatches (`throttle()`), to stay under the gateway's rate limits.
- **Timeouts & retries:** each call gets a 20s timeout (`AbortController`) so one stuck request can't hang the whole batch; failures retry up to 2 more times with backoff (longer for `GatewayRateLimitError` specifically).
- **Output:** live progress (`[N/162] predicted_intent`), a console summary (overall + per-intent accuracy), and `results/jev_predictions.csv` with one row per sample plus an accuracy summary appended as `#`-prefixed comment lines.

**Result:** 98.1% accuracy (158/161 scored, 1 network error) on the stratified eval sample — within two points of the fully fine-tuned RoBERTa, achieved with zero training and a handful of examples per intent.

## Comparison (same 162-row stratified sample)

| | RoBERTa (fine-tuned) | Jev (few-shot) |
|---|---|---|
| Accuracy on the sample | **100%** (162/162) | 98.1% (158/161 scored, 1 error) |
| Accuracy on full test set (818) | 99.88% (817/818) | not run (sample only) |
| Training | 255.5 s on a laptop GPU, ~6.5k labeled examples | none — 3 examples per intent in the prompt |
| Inference | ~1.7 ms/sample, local | network call per row: hundreds of ms to 15 s+ observed, occasionally rate-limited |
| Cost / infra | one-time GPU training, model file to host | per-request API cost, AI Gateway key |

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
dataset/examples.csv            Few-shot example pool (27 intents, ~225-270 rows each)
dataset/eval.csv                Held-out evaluation set
classify.mts                    Few-shot intent classification via Jev / AI Gateway
results/jev_predictions.csv     Per-row Jev predictions + accuracy summary (generated)
```

## Takeaways

- Jev's `choice` evaluation type is a near-perfect fit for intent classification: no separate few-shot API, but a hand-built `criteria` description per label does the job.
- Example *diversity* mattered more than example *count* — de-duplicating near-identical few-shot examples per intent improved grounding without adding more of them.
- The AI Gateway backend's per-request latency is inconsistent (hundreds of ms to 15s+ observed) and occasionally rate-limits under concurrency — a client-side timeout, retry-with-backoff, and request throttle are necessary for a reliable batch run, not optional polish.
- On this dataset, supervised fine-tuning wins on accuracy (100% vs 98.1% on the shared sample; 99.88% on the full test set) and by orders of magnitude on inference speed, but it needs labeled data and a training run. Jev gets within ~2 points with no training and 3 examples per intent, at the cost of per-request latency and API spend.
- The remaining Jev misses are between near-synonymous intents (e.g. `check_invoice` vs `get_invoice`) — a labeling-boundary problem that fine-tuning on thousands of examples learns and a few-shot prompt does not.
