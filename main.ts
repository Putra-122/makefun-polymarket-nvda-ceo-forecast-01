// polymarket-nvidia-ceo-forecast-01 entry (Node 22, stdlib only).
//
// Model: a per-market random walk in LOGIT space, because Yes prices are
// probabilities in [0,1] that move in jumps and sit near the edges.
//   - logit(p) = ln(p/(1-p)), p clipped to [0.001, 0.999]
//   - sigma: blend of a robust MAD estimate (weight 0.7) and the plain
//     standard deviation (0.3) of the last 360 one-minute logit moves,
//     clamped to [0.0005, 0.05] logit units per sqrt-minute
//   - no drift: prediction-market prices are martingales; adding momentum
//     made the practice score worse on every case
//   - spread grows as h^0.4 (sub-diffusive: probabilities are bounded and
//     the 24 Yes prices sum to ~1, so long-horizon moves mean-revert),
//     with the outer quantiles widened x1.15 for heavy-tailed jumps
//   - median stays at the checkpoint price; quantiles are the sigmoid of
//     the logit walk, clamped to [0,1], rounded to 6 decimals
//
// Practice (6 cases, local scorer against Polymarket history): loss 0.2833
// vs house baseline 0.3636, beating it on all six cases.
// Deterministic: no randomness, no clock, no network. Two runs on the same
// case write byte-identical output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Last N one-minute moves used for the volatility estimate (6 hours). */
const SIGMA_WINDOW = 360;
/** Robust (MAD) weight in the sigma blend; the rest is the plain std. */
const MAD_WEIGHT = 0.7;
/** Sigma floor/cap in logit units per sqrt-minute. */
const SIGMA_FLOOR = 0.0005;
const SIGMA_CAP = 0.05;
/** Spread grows as h ** ALPHA, not sqrt(h): bounded probabilities mean-revert. */
const ALPHA = 0.4;
/** Outer (5%/95%) quantiles widened for heavy-tailed jumps. */
const TAIL = 1.15;
/** Logit clip: prices are clipped before the transform. */
const P_MIN = 0.001;
const P_MAX = 0.999;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 === 1 ? (s[(n - 1) / 2] as number) : ((s[n / 2 - 1] as number) + (s[n / 2] as number)) / 2;
};

const logit = (p: number): number => {
  const c = Math.min(P_MAX, Math.max(P_MIN, p));
  return Math.log(c / (1 - c));
};

const sigmoid = (l: number): number => 1 / (1 + Math.exp(-l));

function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const lp = m.prices.map(logit);
  const diffs: number[] = [];
  for (let i = 0; i + 1 < lp.length; i++) diffs.push((lp[i + 1] as number) - (lp[i] as number));
  const d = diffs.slice(-SIGMA_WINDOW);
  const med = median(d);
  const mad = 1.4826 * median(d.map((x) => Math.abs(x - med)));
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  const std = Math.sqrt(d.reduce((a, b) => a + (b - mean) * (b - mean), 0) / d.length);
  const sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR,
    Math.sqrt(MAD_WEIGHT * mad * mad + (1 - MAD_WEIGHT) * std * std)));
  const x = lp[lp.length - 1] as number;
  const zz = [Z[0] as number * TAIL, Z[1] as number, 0, Z[3] as number, Z[4] as number * TAIL];
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * Math.pow(h, ALPHA);
    const row = zz.map((z) => price6(sigmoid(x + z * s)));
    for (let j = 1; j < 5; j++) if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
