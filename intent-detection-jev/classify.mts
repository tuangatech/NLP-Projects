// Zero-shot / few-shot intent classification using Jev (typesafe-ai/jev) via Vercel AI Gateway.
// Compares Jev's picks against the labeled intents in dataset/data_test.csv, as a baseline
// against the fine-tuned RoBERTa model in the notebook.
import { experimental_evaluate as evaluate } from 'ai';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { parse } from 'csv-parse/sync';

const MODEL = 'typesafe-ai/jev';
const SAMPLE_PER_INTENT = 6; // ~162 rows across 27 intents
const EXAMPLES_PER_INTENT = 3; // few-shot examples baked into each choice's description
const CONCURRENCY = 3;
const REQUEST_TIMEOUT_MS = 20_000; // Jev/AI Gateway latency is inconsistent; don't let one call hang the batch
const MIN_REQUEST_INTERVAL_MS = 300; // stagger request starts across all workers to avoid bursts
const MAX_RETRIES = 2; // extra attempts beyond the first, backed off to ride out rate limits
const RATE_LIMIT_BACKOFF_MS = 3000; // base backoff before retrying a rate-limited request
const OTHER_ERROR_BACKOFF_MS = 500; // base backoff before retrying any other error

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Global stagger: every worker waits its turn here before dispatching, so CONCURRENCY workers
// don't all fire at the same instant and trip the gateway's rate limiter.
let nextDispatchAt = Date.now();
async function throttle(): Promise<void> {
  const now = Date.now();
  const wait = Math.max(0, nextDispatchAt - now);
  nextDispatchAt = Math.max(now, nextDispatchAt) + MIN_REQUEST_INTERVAL_MS;
  if (wait > 0) await sleep(wait);
}

type Row = { utterance: string; intent: string; category: string; tags: string };

function loadCsv(path: string): Row[] {
  return parse(readFileSync(path, 'utf-8'), {
    columns: true,
    skip_empty_lines: true,
  });
}

function groupByIntent(rows: Row[]): Map<string, Row[]> {
  const groups = new Map<string, Row[]>();
  for (const row of rows) {
    if (!groups.has(row.intent)) groups.set(row.intent, []);
    groups.get(row.intent)!.push(row);
  }
  return groups;
}

function tokenize(text: string): Set<string> {
  return new Set(text.toLowerCase().match(/[a-z0-9']+/g) ?? []);
}

function jaccard(a: Set<string>, b: Set<string>): number {
  let intersection = 0;
  for (const token of a) if (b.has(token)) intersection += 1;
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

// Greedy farthest-point selection: start with the first utterance, then repeatedly add
// whichever remaining candidate is least similar (by word overlap) to everything picked so
// far. Avoids near-duplicate few-shot examples like "cancelling order" / "cancel order".
function pickDiverseExamples(utterances: string[], n: number): string[] {
  if (utterances.length <= n) return utterances;
  const tokenSets = utterances.map(tokenize);
  const pickedIndices = [0];
  while (pickedIndices.length < n) {
    let bestIndex = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < utterances.length; i++) {
      if (pickedIndices.includes(i)) continue;
      const maxSimToPicked = Math.max(...pickedIndices.map((p) => jaccard(tokenSets[i], tokenSets[p])));
      const score = -maxSimToPicked; // lower similarity to picked = better
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }
    pickedIndices.push(bestIndex);
  }
  return pickedIndices.map((i) => utterances[i]);
}

async function runWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: limit }, worker));
  return results;
}

const trainRows = loadCsv('dataset/data_train.csv');
const testRows = loadCsv('dataset/data_test.csv');

const trainByIntent = groupByIntent(trainRows);
const testByIntent = groupByIntent(testRows);

// Few-shot criteria: each intent's description carries a handful of diverse real training
// examples, since Jev's "choice" question has no dedicated few-shot examples field.
const criteria: Record<string, string> = {};
for (const [intent, rows] of trainByIntent) {
  const diverse = pickDiverseExamples(rows.map((r) => r.utterance), EXAMPLES_PER_INTENT);
  criteria[intent] = `e.g., ${diverse.map((u) => `"${u}"`).join('; ')}`;
}
const intents = Object.keys(criteria);

console.log('\n--- Jev criteria (sent as the "choice" question on every request) ---');
console.log(JSON.stringify(criteria, null, 2));
console.log('--- end criteria ---\n');

// Stratified sample of the test set: first N rows per intent, deterministic across runs.
const sample: Row[] = [];
for (const rows of testByIntent.values()) {
  sample.push(...rows.slice(0, SAMPLE_PER_INTENT));
}

console.log(`Loaded ${intents.length} intents, ${trainRows.length} train rows.`);
console.log(`Evaluating ${sample.length} sampled test rows with ${MODEL}...`);

type Result = {
  utterance: string;
  expected_intent: string;
  predicted: string | null;
  confidence: number | null;
  correct: boolean;
  error: string | null;
};

async function classifyOnce(row: Row): Promise<Result> {
  await throttle();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const { answers } = await evaluate({
      model: MODEL,
      state: row.utterance,
      questions: {
        intent: {
          type: 'choice',
          instructions:
            'Classify this customer support message into the single best matching intent category.',
          criteria,
        },
      },
      abortSignal: controller.signal,
    });
    const answer = answers.intent as { choice: string; probabilities: Record<string, number> };
    return {
      utterance: row.utterance,
      expected_intent: row.intent,
      predicted: answer.choice,
      confidence: answer.probabilities[answer.choice] ?? null,
      correct: answer.choice === row.intent,
      error: null,
    };
  } catch (err) {
    const message = controller.signal.aborted
      ? `timed out after ${REQUEST_TIMEOUT_MS}ms`
      : err instanceof Error
        ? err.message
        : String(err);
    return {
      utterance: row.utterance,
      expected_intent: row.intent,
      predicted: null,
      confidence: null,
      correct: false,
      error: message,
    };
  } finally {
    clearTimeout(timeout);
  }
}

let completed = 0;
const results = await runWithConcurrency(sample, CONCURRENCY, async (row): Promise<Result> => {
  let result = await classifyOnce(row);
  let attempts = 1;
  while (result.error && attempts <= MAX_RETRIES) {
    const isRateLimit = /RateLimit/i.test(result.error);
    const backoff = (isRateLimit ? RATE_LIMIT_BACKOFF_MS : OTHER_ERROR_BACKOFF_MS) * attempts;
    await sleep(backoff);
    result = await classifyOnce(row);
    attempts += 1;
  }
  completed += 1;
  const retriedTag = attempts > 1 ? ` [attempt ${attempts}]` : '';
  const outcome = result.error
    ? `ERROR (${result.error})${retriedTag}`
    : `${result.predicted}${result.correct ? '' : ` (expected: ${result.expected_intent})`}${retriedTag}`;
  console.log(`[${completed}/${sample.length}] ${outcome}`);
  return result;
});

const errors = results.filter((r) => r.error);
const scored = results.filter((r) => !r.error);
const accuracy = scored.length ? scored.filter((r) => r.correct).length / scored.length : 0;

console.log(`\nOverall accuracy: ${(accuracy * 100).toFixed(1)}% (${scored.length} scored, ${errors.length} errored)`);

const perIntent = new Map<string, { correct: number; total: number }>();
for (const r of scored) {
  const stat = perIntent.get(r.expected_intent) ?? { correct: 0, total: 0 };
  stat.total += 1;
  if (r.correct) stat.correct += 1;
  perIntent.set(r.expected_intent, stat);
}
console.log('\nPer-intent accuracy:');
for (const [intent, stat] of [...perIntent].sort((a, b) => a[1].correct / a[1].total - b[1].correct / b[1].total)) {
  console.log(`  ${intent.padEnd(28)} ${stat.correct}/${stat.total}`);
}

mkdirSync('results', { recursive: true });
const header = 'utterance,expected_intent,predicted,confidence,correct,error\n';
const csvEscape = (s: string) => `"${s.replace(/"/g, '""')}"`;
const lines = results.map((r) =>
  [
    csvEscape(r.utterance),
    r.expected_intent,
    r.predicted ?? '',
    r.confidence ?? '',
    r.correct,
    r.error ? csvEscape(r.error) : '',
  ].join(','),
);
const summary = [
  '',
  `# accuracy,${accuracy.toFixed(4)}`,
  `# scored,${scored.length}`,
  `# errored,${errors.length}`,
  `# total,${results.length}`,
].join('\n');
writeFileSync('results/jev_predictions.csv', header + lines.join('\n') + '\n' + summary + '\n');
console.log('\nWrote results/jev_predictions.csv');
