# The walk-forward optimizer

The optimizer's vectorized engine (`solopt/`) runs from two places, on two
schedules, and there is no PC and no git hand-off repository in the loop
anywhere:

- **Daily**, in the trading worker's own process, on the droplet's own CPU —
  a *focused* search narrowed to the neighbourhood of the last accepted
  parameter set, cheap enough to finish in about 5-10 minutes without
  meaningfully competing with the trading loop for the vCPU.
- **Monthly**, on a GPU worker rented from RunPod for the job's duration only —
  a full sweep across the entire parameter space, wide enough to catch
  anything the narrowed daily search misses.

```
┌───────────────────────────┐                     ┌───────────────────────────┐
│         droplet            │                     │      RunPod GPU worker     │
│                            │                     │    (rented per job, one    │
│  solbot-worker              │                     │     or more per batch)     │
│  ├─ daily: solopt.pipeline ─┼─ in-process, CPU     │                            │
│  │  (focused grid)          │                     │                            │
│  └─ monthly: solbot.wfmc ────┼── ship Parquet ────▶│  solopt.cli run            │
│     .run_monthly()          │   + config           │  (full grid, CuPy/NumPy)   │
│     solbot.runpod           │◀── feed + bundle ────┤  solopt.report pushes back │
│     .RunPodClient           │   over HTTP           │                            │
│                            │── terminate + ───────▶│  (torn down, verified)     │
│  SQLite: optimizer_runs,    │   verify teardown     └───────────────────────────┘
│  param_bundles, settings    │
│  _audit, WF/MC results      │
└───────────────────────────┘
```

A finished bundle goes straight into `solbot.paramsync.accept_bundle()` —
directly, in-process, for the daily run; over the authenticated
`/optimizer/bundle` endpoint, for a RunPod worker reporting back. Either way it
is re-checked against the same bounds the settings page enforces before it is
allowed anywhere near shadow. Parameter history now lives in `param_bundles`
and `settings_audit`, not in a git log.

---

## Setup

Nothing to install anywhere else — the daily run is already part of
`solbot-worker` and needs no configuration beyond the defaults. The monthly
RunPod run needs a RunPod account and its own credentials:

1. **RunPod API key.** From the RunPod console, put it in `.env` as
   `RUNPOD_API_KEY`.
2. **RunPod S3-compatible network volume credentials** (for shipping the
   candle bundle to the worker): `RUNPOD_S3_ACCESS_KEY` / `RUNPOD_S3_SECRET_KEY`.
3. On the settings page, set `runpod_callback_url` to this droplet's own public
   URL (so a worker knows where to report back to), and generate a **RunPod
   worker token**: Settings → *RunPod worker token* → *Generate a new token*.
   It grants write access to the run feed and bundle submission only — it
   cannot halt trading, close a position, or read an API key.
4. Turn on `wfmc_monthly_enabled` once the above is in place. `wfmc_daily_enabled`
   is on by default and needs none of this.

`runpod_gpu_type` and `runpod_batch_size` (coins per parallel job) are tunable
from the same settings group.

---

## Running it

Both runs are driven by the worker's own schedule (`wfmc_daily_hour_utc`,
`wfmc_monthly_day_utc` / `wfmc_monthly_hour_utc`), or on demand from the
walk-forward tab's "Run daily walk-forward now" / "Run monthly RunPod retest
now" buttons — there is no CLI step for an operator to run.

`python -m solopt run` still exists, but it is the RunPod worker's own entry
point, not something you invoke by hand: `solbot.runpod.RunPodClient.launch_batch()`
starts the worker's container with the droplet URL, worker token, a report run
id, and the search config baked into its environment, and the container runs
`solopt.cli.cmd_run()` against the Parquet chunk it was shipped, then pushes
its feed, run summary and finished bundle back over HTTP via `solopt/report.py`.

### Interrupting a run

A run is checkpointed **per window** in its own local `RunStore`, same as
before. A daily run that gets killed mid-way (worker restart, e.g.) simply
picks the search back up at the next scheduled run rather than resuming — the
focused grid is cheap enough that this costs minutes, not hours. A RunPod
worker that dies mid-batch is caught by teardown verification (below) rather
than silently leaving a bill running.

### The utilization slider

`utilization_pct` in the walk-forward config caps how much wall-clock time the
search may use — at 60 it works for 60ms out of every 100ms and sleeps the
rest. On the daily run this is what keeps it from meaningfully competing with
the trading loop and live fuzzy-regime classification for the droplet's own
vCPUs (see `deploy/DEPLOY.md` for the current droplet size); on a RunPod
worker it matters far less, since nothing else is sharing that GPU.

### Teardown verification

After a monthly run's batches all report back (or time out), the droplet calls
`RunPodClient.verify_teardown()` for every job it launched: it terminates each
pod and network volume, then re-checks that nothing is still running and
force-stops anything that did not self-terminate. The result — clean or not —
is logged to the event feed either way, so "did the GPU actually stop billing"
is never a question you have to go check the RunPod console to answer.
`tests/test_runpod.py` pins this against a fake transport; no test here ever
makes a live RunPod call.

---

## What it actually does

### 1. The vectorized engine

Signals and trade outcomes are computed as batched array operations across every
symbol and every parameter combination at once. There is no per-combination loop
over candles anywhere in it.

Four stages: combinations are grouped by indicator signature so combinations
sharing every *period* share every *series*; entry masks are built by
broadcasting per-combination thresholds against those shared arrays; the sparse
mask collapses to a flat candidate list and exits are resolved by a bounded
forward scan (`H` array operations, `H` being the maximum hold in bars); and
finally a sequential sweep applies the concurrency and sizing caps, walking the
few thousand surviving candidates rather than the millions of bars.

Measured on one CPU core with NumPy: 96 combinations × 60 symbols × 8,000 bars in
3.7 seconds, or about 46 million combination-bar-symbol evaluations.

`tests/test_vector_parity.py` asserts this engine reproduces the shipped
`solbot.backtest.Backtester` **trade for trade** on identical data, and that
every indicator matches the live pandas implementation bar for bar. That is not
a nicety: the moment the two diverge, every number the optimizer produces is a
statement about a strategy nobody is running.

### 2. Walk-forward

Tune on an in-sample window, test on the next window the search has never seen,
roll forward, repeat. The thresholds are unforgiving on purpose:

- **30 closed trades** before a window counts at all. A window that fired six
  trades is a coin flip wearing a percentage sign.
- **70% of counted windows profitable** out of sample.
- **OVERFIT** if the standard deviation of window returns exceeds their mean. A
  strategy that averages 2% by making 20% once and losing 6% four times is a
  lottery ticket with a backtest.

The search itself stops at an iteration cap **or** early once the best result
plateaus. The plateau rule matters more: a search that keeps going after the
objective flattens is not finding a better strategy, it is finding a luckier
arrangement of the same noise.

The objective is return per unit of drawdown, discounted for churn. Raw return
picks the set that got luckiest; return over drawdown picks the set an operator
could actually hold; and the churn discount makes the search *pay* for trade
frequency, which is what stops it rediscovering the 35-trades-a-day behaviour
that produced the earlier fee bill.

**Per-coin parameters** are tuned only where a coin has enough of its own trade
history to support them (60 by default). A coin with nine trades falls back to
the global set, which is the honest answer — giving it bespoke parameters anyway
is how a walk-forward run quietly becomes per-coin curve fitting.

### 3. Monte Carlo

Not another backtest. It consumes exactly the out-of-sample trades the
walk-forward run already produced, and asks a different question: how much of
that was the order things happened in, and how much survives execution behaving
as badly as it sometimes does?

Slippage and fees are sampled per trade from the droplet's **observed** fills
(`/api/optimizer/execution`), not from the constants the backtest assumed. Below
30 recorded fills it says so and falls back to the constants — a run against
assumed costs is a weaker claim than one against measured ones, and the
dashboard shows which you got.

> **One deliberate departure from the spec.** The spec asks for randomised trade
> *order*. Under fixed-fraction sizing the equity path is a cumulative product,
> and a product does not care what order its terms come in — so permutation alone
> gives many drawdown paths and exactly *one* terminal value. A "5th-percentile
> terminal return" built that way would report the spread of the execution noise
> and nothing else. The default therefore draws trades **with replacement** (the
> standard bootstrap), which keeps the ordering randomness and adds the sampling
> question that makes the tail mean something. `resample: "permutation"` restores
> the literal reading if you want the drawdown-only view.

The headline output is the **5%-worst-case maximum drawdown**, and it is what
feeds position sizing — not the historical backtest drawdown, which is one draw
from this distribution with no claim to being the representative one.

**Walk-forward efficiency** is reported alongside it: out-of-sample return per
day divided by in-sample return per day. The per-day normalisation matters,
because in-sample windows are four times longer here and comparing raw totals
would report roughly 0.25 for a strategy that generalised perfectly.

### 4. Crash replay

Walk-forward windows roll evenly, which means the worst days get averaged in
with everything else. A set that loses 40% in the two days the market fell apart
and makes it back over six weeks passes every threshold above and is still a set
nobody should run.

So the crash windows are pulled out and replayed on their own. They are **found
in the data**, not hard-coded to a list of dates: an equal-weight index is built
from the panel, the deepest sustained declines and the sharpest single-bar drops
become the stress windows. A set that fails one is flagged **FRAGILE**, and a
fragile flag blocks auto-promotion however good the ordinary windows looked.

Refusing to trade through a crash counts as a pass, not a failure to measure —
the regime gate doing its job looks exactly like that.

---

## Survivorship bias

The historical test universe is rebuilt from `universe_snapshots.csv`: what
actually passed the liquidity and volume floors **on each historical day**,
including the coins that later died. A coin that rugged in month two is eligible
in month one and not afterwards, which is precisely the population the live bot
was choosing from at the time.

If the bundle has no snapshot file the loader says so loudly and treats every
symbol as eligible throughout — which reintroduces the bias. Pull snapshots.

---

## Where results land

An accepted run's bundle carries its own evidence: the walk-forward summary,
the Monte Carlo distribution, and every stress window's result travel with the
parameters, straight into `solbot.paramsync.accept_bundle()` — in-process for
the daily run, over the bearer-token-authenticated `/optimizer/bundle` endpoint
for a RunPod worker reporting back.

That is what lets the droplet refuse a bundle **on its own terms**, whichever
side produced it. It re-checks the gates rather than believing them, validates
every parameter against the same bounds the settings page enforces, and drops
anything that fails. A RunPod worker's output is another machine's claim, not
an authority — the daily run gets exactly the same scrutiny even though it
never left this process.

A bundle that passes lands in the **shadow** instance — paper trading alongside
the real one, on the same data, through the same executor. Promotion to live is
automatic and needs no approval, but it needs:

- 15 clean days in shadow,
- at least 30 closed shadow trades,
- shadow beating live on expectancy by the configured margin (25% by default),
- a drawdown not materially deeper than live's,
- and a bootstrap comparison of the two trade populations where shadow wins at
  least 90% of resampled draws.

That last one is what turns "a decisive margin, not a marginal edge" into
something the code can check. Comparing two averages says which is bigger; it
does not say whether the difference would survive a different run of the same
two strategies.

Every promotion decision is recorded, including the refusals.

---

## Adding another asset class

The engine works on "symbols" and "bars". It has no idea whether a symbol is a
Solana mint, a NYSE ticker or a currency pair.

Adding equities or forex means adding a `Schema` in `solopt/schema.py` and a
loader that emits the canonical `ts, open, high, low, close, volume` columns.
`EQUITY` and `FOREX` schemas already exist so the generic path is exercised
rather than merely claimed. The one thing that genuinely cannot be abstracted is
the cost model — a crypto AMM charges a percentage and slips against pool depth,
an equity broker charges per share against a spread — so `CostModel` carries all
three terms and each schema fills in the ones that apply.
