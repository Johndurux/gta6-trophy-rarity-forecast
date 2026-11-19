#!/usr/bin/env node
// gta6-trophy-rarity-forecast-01 — quantile forecast of every trophy's Exophase earn rate
// at the case's target, Node 22, standard library only, no network.
//
// What a rate is: 100 x (earners of the trophy) / (players Exophase tracks for the game). Both
// parts move. The denominator (players_tracked) grows as more people sync their profile; the
// numerator grows as players reach the trophy. Forecasting the rate therefore means forecasting
// two curves on the case's own history and taking their ratio, never carrying the last rate
// forward unchanged: on launch night the page was frozen for hours while both parts moved
// underneath it, and over a week the story trophies gain earners far faster than the tracked
// base grows.
//
// The model:
//   1. denominator: players_tracked against the hours since the case's first read, a power law
//      (least squares on log-log, one robust rejection pass), extrapolated to the target and
//      clamped so it never falls below the last read;
//   2. numerator: for each trophy its earners (rate x tracked / 100) against the same hours,
//      a power law per trophy, shrunk toward the pooled slope of all trophies so that a trophy
//      with four earners does not invent its own trend from rounding noise;
//   3. a stale page: when the recording has not changed for hours the rates understate what the
//      next refresh will show, so the gap is carried forward with the growth the case itself
//      measured (or a launch-week prior when it measured none);
//   4. quantiles: a two-component mixture in logit space, wide enough for a four-day horizon,
//      slightly skewed down because the observed failure mode of a launch-week forecaster is
//      an under-estimated denominator.
// The story chain (Complete Chapter 1..n, the final mission) is made non-increasing with a
// pool-adjacent-violators pass: the funnel is a hard fact of the game, not a fitted parameter.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, matching the house baseline's z. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Milliseconds in an hour. */
const HOUR = 3_600_000;

/** Spread of the rate's logit per square-root day, at the six-day scale the final asks for. */
const SIGMA_PER_SQRT_DAY = Number(process.env.EVAL_SIGMA ?? 0.78);
/** The two-component mixture's weights and shapes: a core and a wider tail. */
const CORE_WEIGHT = 0.8;
const CORE_SCALE = 0.8;
const TAIL_SCALE = 1.25;
const TAIL_DROP = 0.12;
/** Hours of frozen recording after which the pending refresh is modelled, and the priors it falls back on. */
const DEFAULT_TRACKED_SLOPE = 0.9;
/** Hours of history the slopes are fitted on: the launch week changes regime, so recent reads count more. */
const FIT_HOURS = 48;
/**
 * Per-hour growth of the tracked base while a frozen page waits for its refresh, when the case's
 * own refreshes measured none: the launch week's own curve, decaying as the game ages (a page
 * frozen four hours after launch gains far more per hour than one frozen on day three).
 */
const STALE_TRACKED_GROWTH = Number(process.env.EVAL_STALE_TRACKED ?? 0.386);
/**
 * Anchor of that prior, per hour, taken from the one frozen-to-fresh interval the rules document
 * themselves: the house's page showed 638 players tracked at 2026-11-18T23:50Z and 3,868 at its
 * 04:30Z refresh, ln(3868 / 638) / 4.67 h. It decays with the game's age, so a page frozen on
 * day three is not credited with launch night's hourly growth.
 */
const LAUNCH_AGE_HOURS = 5;
const STALE_EARNER_GROWTH = Number(process.env.EVAL_STALE_EARNER ?? 0.02);
/** Above this rate a trophy is "everyone has it", and its quantiles are read off the ceiling instead. */
const CEILING_RATE = 99.5;
/** Shrinkage length: how many reads a trophy's own slope needs to outweigh the pooled one. */
const SHRINK_READS = 6;

interface Trophy { id: number; name: string; description: string; tier: string; points: number; secret: boolean }
interface Read { t: number; tracked: number; earned: number; rates: Array<number | null> }
interface Case {
  id: string;
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  game: { name?: string; release_date?: string };
  trophies: Trophy[];
  history: Array<{ slot: string; at: string; players_tracked: number; trophies_earned: number; rates: Array<number | null> }>;
}
interface Row { id: number; q: number[] }

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** The scorer's clamp: a rate outside 0.01..99.99 has no logit of its own. */
const rateLogit = (pct: number): number => {
  const c = clamp(pct, 0.01, 99.99);
  return Math.log(c / (100 - c));
};
const expit = (y: number): number => 100 / (1 + Math.exp(-y));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Least squares of y on x, both already transformed: slope, intercept, residual sd, points used. */
function linfit(xs: number[], ys: number[]): { b: number; a: number; sd: number; n: number } {
  const n = xs.length;
  if (n < 2) return { b: 0, a: n === 1 ? ys[0]! - 0 * xs[0]! : 0, sd: 0, n };
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]!; sy += ys[i]!; }
  const mx = sx / n, my = sy / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i]! - mx; sxx += dx * dx; sxy += dx * (ys[i]! - my); }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) { const r = ys[i]! - (a + b * xs[i]!); ss += r * r; }
  return { b, a, sd: n > 2 ? Math.sqrt(ss / (n - 2)) : 0, n };
}

/** A power law logged: slope on (ln hours, ln value), one pass rejecting points over 3 sd. */
function powerSlope(xs: number[], ys: number[]): { b: number; n: number; sd: number } {
  let keep = xs.map((_, i) => i);
  let fit = linfit(keep.map((i) => xs[i]!), keep.map((i) => ys[i]!));
  if (fit.n >= 5 && fit.sd > 1e-9) {
    const kept: number[] = [];
    for (const i of keep) {
      const r = Math.abs(ys[i]! - (fit.a + fit.b * xs[i]!));
      if (r <= 3 * fit.sd) kept.push(i);
    }
    if (kept.length >= 3 && kept.length < keep.length) {
      keep = kept;
      fit = linfit(keep.map((i) => xs[i]!), keep.map((i) => ys[i]!));
    }
  }
  return { b: fit.b, n: fit.n, sd: fit.sd };
}

/** Pool-adjacent-violators: the non-increasing fit of a sequence, positions kept. */
function isotonicDown(values: number[]): number[] {
  // values in the order the chain runs; the result is non-increasing in that order.
  const blocks: Array<{ sum: number; n: number }> = [];
  for (const v of values) {
    blocks.push({ sum: v, n: 1 });
    // merge right-to-left while the block to the left is below its right neighbour
    while (blocks.length > 1) {
      const r = blocks[blocks.length - 1]!, l = blocks[blocks.length - 2]!;
      if (l.sum / l.n < r.sum / r.n) {
        blocks.splice(blocks.length - 2, 2, { sum: l.sum + r.sum, n: l.n + r.n });
      } else break;
    }
  }
  const out: number[] = [];
  for (const b of blocks) for (let i = 0; i < b.n; i++) out.push(b.sum / b.n);
  return out;
}

/** Quantiles of the two-component logit mixture around a median `m` with base spread `s`. */
function mixtureQuantiles(m: number, s: number, lean = 0): number[] {
  const tail = 1 - CORE_WEIGHT;
  // `lean` turns the tail over: a trophy pinned against the floor can only gain, so its second
  // component sits above the median instead of below it
  const tailAt = lean > 0 ? m + 0.4 * lean * s : m - TAIL_DROP * s;
  const cdf = (y: number): number => {
    const core = 0.5 * (1 + erf((y - m) / (s * CORE_SCALE * Math.SQRT2)));
    const low = 0.5 * (1 + erf((y - tailAt) / (s * TAIL_SCALE * Math.SQRT2)));
    return CORE_WEIGHT * core + tail * low;
  };
  const out: number[] = [];
  for (const tau of LEVELS) {
    let lo = m - 12 * s - 2, hi = m + 12 * s + 2;
    for (let k = 0; k < 80; k++) {
      const mid = (lo + hi) / 2;
      if (cdf(mid) < tau) lo = mid; else hi = mid;
    }
    out.push(round4(clamp(expit((lo + hi) / 2), 0, 100)));
  }
  // non-decreasing, exactly as the scorer requires
  for (let i = 1; i < out.length; i++) if (out[i]! < out[i - 1]!) out[i] = out[i - 1]!;
  return out;
}

/** The error function, Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7): enough for a quantile bisection. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

/** Hours since the case's first read, one hour to keep the log away from zero. */
const hoursOf = (t: number, t0: number): number => (t - t0) / HOUR + 1;

/** Everything the model reads off one case, computed once. */
interface Frame {
  reads: Read[];
  hLast: number;
  hCut: number;
  hTarget: number;
  days: number;
  trackedLast: number;
  trackedSlope: number;
  trackedSd: number;
  /** Hours between the recording's last change and the checkpoint: what the page has not shown yet. */
  backlogHours: number;
  trackedPerHour: number;
  earnerPerHour: number[];
}

/** The case's reads, deduplicated to the changes a page refresh made, and the frame around them. */
function frameOf(c: Case): Frame {
  const reads: Read[] = [];
  for (const r of c.history) {
    const t = Date.parse(r.at ?? r.slot);
    if (!Number.isFinite(t) || !Number.isFinite(r.players_tracked) || r.players_tracked <= 0) continue;
    reads.push({ t, tracked: r.players_tracked, earned: r.trophies_earned, rates: r.rates });
  }
  if (reads.length === 0) throw new Error(`${c.id} has no usable read`);
  reads.sort((a, b) => a.t - b.t);
  const targetT = Date.parse(c.target);
  const cutT = Number.isFinite(Date.parse(c.at)) ? Date.parse(c.at) : reads[reads.length - 1]!.t;
  const t0 = reads[0]!.t;
  const last = reads[reads.length - 1]!;
  const days = c.horizon_minutes / 1440;

  // the denominator: log-log over the reads of the last 60 hours, which is where the launch's
  // growth curve is still being traced
  const from = last.t - FIT_HOURS * HOUR;
  const tx: number[] = [], ty: number[] = [];
  for (const r of reads) {
    if (r.t < from) continue;
    tx.push(Math.log(hoursOf(r.t, t0)));
    ty.push(Math.log(r.tracked));
  }
  const fit = powerSlope(tx, ty);
  // a flat recording carries no growth signal at all: a slope fitted to plateaus is noise
  const distinct = new Set(reads.map((r) => r.tracked)).size;
  const trackedSlope = distinct >= 3 && fit.n >= 3 ? clamp(fit.b, 0, 1.4) : 0;

  // how long the recording has been frozen, and how fast the two parts moved between refreshes:
  // the ratio's own history is the best estimate of what the next refresh will show
  const changeAt = 24 * HOUR;
  let lastChange = reads[0]!.t;
  let dTracked = 0, dtTracked = 0;
  const dEarner: number[] = c.trophies.map(() => 0);
  const dtEarner: number[] = c.trophies.map(() => 0);
  for (let i = 1; i < reads.length; i++) {
    const a = reads[i - 1]!, b = reads[i]!;
    const dt = (b.t - a.t) / HOUR;
    const moved = a.tracked !== b.tracked || a.rates.some((r, j) => r !== b.rates[j]);
    if (!moved) continue;
    lastChange = b.t;
    if (b.t < last.t - changeAt || dt <= 0) continue;
    if (a.tracked > 0 && b.tracked > a.tracked) { dTracked += Math.log(b.tracked / a.tracked); dtTracked += dt; }
    for (let j = 0; j < c.trophies.length; j++) {
      const ra = a.rates[j], rb = b.rates[j];
      if (ra === null || ra === undefined || rb === null || rb === undefined) continue;
      const ea = Math.max(0.6, (ra / 100) * a.tracked), eb = Math.max(0.6, (rb / 100) * b.tracked);
      if (eb <= ea) continue;
      dEarner[j]! += Math.log(eb / ea);
      dtEarner[j]! += dt;
    }
  }
  // a read every ten minutes and a refresh every half hour means half an hour of backlog is normal,
  // not staleness: the backlog starts where the recorder's own cadence ends
  const backlogHours = clamp((cutT - lastChange) / HOUR, 0, 12);

  return {
    reads,
    hLast: hoursOf(last.t, t0),
    hCut: hoursOf(cutT, t0),
    hTarget: hoursOf(targetT, t0),
    days,
    trackedLast: last.tracked,
    trackedSlope,
    trackedSd: distinct >= 3 ? fit.sd : 0,
    backlogHours,
    trackedPerHour: dtTracked > 0
      ? dTracked / dtTracked
      : STALE_TRACKED_GROWTH / (1 + Math.max(0, hoursOf(last.t, t0) - LAUNCH_AGE_HOURS) / 20),
    earnerPerHour: c.trophies.map((_, j) => (dtEarner[j]! > 0 ? dEarner[j]! / dtEarner[j]! : STALE_EARNER_GROWTH)),
  };
}

/** Which extrapolator the medians use: the log-log power law, the absolute flow, or their mean. */
const VARIANT = process.env.EVAL_VARIANT ?? 'power';

/** The median rate of each trophy at the target, in the case's trophy order. */
function medians(c: Case, f: Frame): number[] {
  const power = mediansPower(c, f);
  if (VARIANT === 'power') return power;
  const flow = mediansFlow(c, f);
  if (VARIANT === 'flow') return flow;
  return power.map((p, i) => expit((rateLogit(p) + rateLogit(flow[i]!)) / 2));
}

/**
 * The absolute-flow extrapolator: earners and tracked players each advance at the per-hour rate
 * the last day measured, so the rate is the ratio of two straight lines. The reference read is
 * the last one at least a day old, which averages over the page's plateaus; a frozen page adds
 * the same pending-refresh correction the power law applies.
 */
function mediansFlow(c: Case, f: Frame): number[] {
  const last = f.reads[f.reads.length - 1]!;
  const back = last.t - 24 * HOUR;
  let ref = f.reads[0]!;
  for (const r of f.reads) if (r.t <= back) ref = r;
  const dt = (last.t - ref.t) / HOUR;
  const ahead = Math.max(0, f.hTarget - f.hLast);
  if (dt < 1 || last.tracked <= ref.tracked) {
    // nothing measured: the page has been frozen, so the power law's refresh prior is all we have
    return mediansPower(c, f);
  }
  const perHourT = (last.tracked - ref.tracked) / dt;
  const tracked = Math.max(last.tracked, last.tracked + perHourT * ahead)
    * Math.exp(f.trackedPerHour * f.backlogHours);
  const out: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const rl = latestRate(f, i);
    const rr = i < ref.rates.length ? ref.rates[i] : null;
    let e = rl === null ? 1 : Math.max(0.6, (rl / 100) * last.tracked);
    if (rl !== null && rr !== null && rr !== undefined) {
      const eRef = Math.max(0.6, (rr / 100) * ref.tracked);
      const perHourE = Math.max(0, (e - eRef) / dt);
      e = e + perHourE * ahead;
    }
    e *= Math.exp(f.earnerPerHour[i]! * f.backlogHours);
    const rate = (100 * e) / tracked;
    out.push(clamp(rl !== null && rl >= CEILING_RATE ? Math.max(rate, 99.9) : rate, 0.001, 99.99));
  }
  return out;
}

/** The median rate of each trophy at the target, in the case's trophy order. */
function mediansPower(c: Case, f: Frame): number[] {
  const t0 = f.reads[0]!.t;
  const ratio = f.hTarget / f.hCut;
  // a flat recording still belongs to a launch week: its slope is the week's usual one
  const bTracked = f.trackedSlope > 0.15 ? f.trackedSlope : DEFAULT_TRACKED_SLOPE;

  // denominator: the last read carries the page's backlog, then the fitted curve runs to the target.
  // With no fitted slope at all the whole way is the launch week's prior, measured from the last
  // change the page showed rather than from the read that repeated it.
  const freezeToTarget = f.backlogHours + (f.hTarget - f.hCut);
  const trackedCut = f.trackedLast * Math.exp(f.trackedPerHour * f.backlogHours);
  const tracked = Math.max(trackedCut, f.trackedSlope > 0.15
    ? trackedCut * Math.pow(ratio, bTracked)
    : f.trackedLast * Math.exp(f.trackedPerHour * freezeToTarget));

  // earners at the target, per trophy: a power law shrunk toward the pooled slope
  const from = f.reads[f.reads.length - 1]!.t - FIT_HOURS * HOUR;
  const slopes: Array<{ b: number; n: number; eLast: number }> = [];
  const xsAll: number[] = [], ysAll: number[] = [], wAll: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const xs: number[] = [], ys: number[] = [];
    for (const r of f.reads) {
      if (r.t < from) continue;
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      const e = Math.max(0.6, (rate / 100) * r.tracked); // six tenths of an earner stands in for "a few"
      xs.push(Math.log(hoursOf(r.t, t0)));
      ys.push(Math.log(e));
    }
    const fit = powerSlope(xs, ys);
    let eLast = 0.6;
    const lastRate = f.reads[f.reads.length - 1]!.rates[i];
    for (let k = f.reads.length - 1; k >= 0; k--) {
      const rate = f.reads[k]!.rates[i];
      if (rate !== null && rate !== undefined) { eLast = Math.max(0.6, (rate / 100) * f.reads[k]!.tracked); break; }
    }
    const n = xs.length >= 2 ? fit.n : 0;
    slopes.push({ b: n >= 2 ? fit.b : NaN, n, eLast });
    if (n >= 2 && Number.isFinite(fit.b)) {
      for (let k = 0; k < xs.length; k++) { xsAll.push(xs[k]!); ysAll.push(ys[k]!); }
      wAll.push(Math.sqrt(eLast));
    }
    void lastRate;
  }
  // pooled slope: the weighted median of the trophies' own slopes, so one odd trophy cannot move it
  const pairs = slopes
    .map((s, i) => ({ b: s.b, w: wAll[Math.min(i, wAll.length - 1)] ?? 1 }))
    .filter((p) => Number.isFinite(p.b));
  pairs.sort((a, b) => a.b - b.b);
  let pool = 0;
  if (pairs.length > 0) {
    const total = pairs.reduce((s, p) => s + p.w, 0);
    let acc = 0;
    pool = pairs[pairs.length - 1]!.b;
    for (const p of pairs) { acc += p.w; if (acc >= total / 2) { pool = p.b; break; } }
  }
  pool = clamp(pool, -0.6, 2.2);

  const out: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const s = slopes[i]!;
    let beta = pool;
    if (Number.isFinite(s.b) && s.n >= 2) {
      beta = (s.n * s.b + SHRINK_READS * pool) / (s.n + SHRINK_READS);
      beta = clamp(beta, pool - 1.1, pool + 1.1);
    }
    beta = clamp(beta, -0.8, 2.6);
    let e = s.eLast * Math.exp(f.earnerPerHour[i]! * f.backlogHours);
    e *= Math.pow(ratio, beta);
    let rate = (100 * e) / tracked;
    // the trophy everyone has stays where it is: a 100% prologue cannot lose its crowd in a week
    const lastRate = latestRate(f, i);
    if (lastRate !== null && lastRate >= CEILING_RATE) rate = Math.max(rate, 99.9);
    out.push(clamp(rate, 0.001, 99.99));
  }
  return out;
}

/** The trophy's rate in the latest read that carried it. */
function latestRate(f: Frame, i: number): number | null {
  for (let k = f.reads.length - 1; k >= 0; k--) {
    const r = f.reads[k]!.rates[i];
    if (r !== null && r !== undefined) return r;
  }
  return null;
}

/** The story chain of a case: the ids of Complete Chapter 1..n and the final mission, in order. */
function storyChain(trophies: Trophy[]): number[] {
  const chapter: Array<{ n: number; i: number }> = [];
  let final = -1;
  trophies.forEach((t, i) => {
    const m = /complete chapter (\d+)/i.exec(t.description ?? '');
    if (m) chapter.push({ n: Number(m[1]), i });
    if (/complete the final mission/i.test(t.description ?? '')) final = i;
  });
  chapter.sort((a, b) => a.n - b.n);
  const chain = chapter.map((c) => c.i);
  if (final >= 0) chain.push(final);
  return chain;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
const outputs: Array<{ file: string; rows: Row[] }> = [];
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const f = frameOf(c);
  const m = medians(c, f);
  // the funnel: the story chain's rates never rise as the story runs on
  const chain = storyChain(c.trophies);
  if (chain.length >= 2) {
    const fitted = isotonicDown(chain.map((i) => rateLogit(m[i]!)));
    chain.forEach((i, k) => { m[i] = expit(fitted[k]!); });
  }
  const s = SIGMA_PER_SQRT_DAY * Math.sqrt(Math.max(f.days, 1 / 48))
    + Number(process.env.EVAL_STALE_SIGMA ?? 0.06) * Math.sqrt(f.backlogHours);
  const rows: Row[] = c.trophies.map((t, i) => {
    const last = latestRate(f, i);
    // a trophy at the ceiling cannot go up: its five quantiles sit in the last hundredth of a percent
    if (last !== null && last >= CEILING_RATE) return { id: t.id, q: [99.9, 99.96, 99.99, 99.99, 99.99] };
    // a rate against the floor has far more room to rise than to fall: its tail leans up
    const lean = m[i]! < 0.05 ? 1 : m[i]! < 0.15 ? 0.5 : 0;
    return { id: t.id, q: mixtureQuantiles(rateLogit(m[i]!), s, lean) };
  });
  outputs.push({ file: `${id}.out`, rows });
  void Z;
}
for (const o of outputs) writeFileSync(join(OUTPUT_DIR, o.file), `${JSON.stringify({ trophies: o.rows })}\n`);