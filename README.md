# Solana TA Trading Bot

A Solana trading bot that trades on pure technical analysis, starts in paper
mode, and is built to survive rather than to chase outsized daily returns.

Standalone: its own directory, its own virtualenv, its own systemd units. It
shares nothing with any other bot on the droplet.

```
┌─────────────────┐         ┌──────────────────┐
│  solbot-worker  │         │    solbot-web    │
│  trading loop   │         │  gunicorn+nginx  │
│  (only process  │         │   dashboard      │
│   that trades)  │         │  (read-mostly)   │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │      ┌──────────────┐     │
         └─────▶│  SQLite (WAL)│◀────┘
                └──────────────┘
      writes state          writes commands
      the web reads         the worker drains
```

The dashboard never opens or closes a position. It enqueues a row in the
`commands` table that the worker executes on its next cycle. One writer on the
trading path, so a dashboard crash cannot touch an open position.

### Walk-forward / Monte Carlo: on the droplet, not on a PC

A focused daily re-score runs on the droplet's own CPU, inside the trading
worker's own process, during a low-activity window (~5-10 minutes). Once a
month, a full parameter sweep runs on a GPU worker rented from RunPod for the
duration of the job: the droplet ships it the candle data it needs, pulls the
results back, and tears the worker down — verified afterward, not assumed.

```
┌─────────────────────┐                          ┌──────────────────────┐
│       droplet        │                          │  RunPod GPU worker    │
│   solbot-worker       │   daily, in-process      │  (monthly only,       │
│   walk-forward        │   CPU, ~5-10 min/day     │   rented per job)     │
│   Monte Carlo         │                          │                       │
│   crash replay        │   monthly: ship data ───▶│  solopt engine        │
│                       │◀──── pull results ───────┤  (CuPy/NumPy)         │
│   SQLite: trades,     │   teardown verified       └──────────────────────┘
│   settings, audit,    │
│   WF/MC results       │
└─────────────────────┘
```

A parameter bundle that passes lands straight in SQLite's shadow-instance
tables — re-checked against the same bounds the settings page enforces, never
just trusted. There is no git hand-off repository and no operator PC in the
loop; `solopt`'s vectorized engine does not change between the two runs, only
where it is invoked from. Full detail in [docs/OPTIMIZER.md](docs/OPTIMIZER.md).

---

## Quick start (paper mode)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Fill in `.env` (see below), then:

```bash
python manage.py init-db && python manage.py create-user yourname
```

```bash
python manage.py check-apis
```

Run the two processes:

```bash
python run_worker.py
```

```bash
python wsgi.py
```

Dashboard at <http://127.0.0.1:8090>. For local HTTP set
`SOLBOT_INSECURE_COOKIES=1`, otherwise the session cookie is `Secure` and will
not survive a plain-HTTP login.

For the droplet, see [deploy/DEPLOY.md](deploy/DEPLOY.md).

---

## What it does

**Universe.** Binance.US's top ~100 coins by 24h quote volume (`binance_top_n`),
intersected with what is actually routable on Solana via Jupiter — established
coins, not a pure liquidity/volume floor over Jupiter's entire token list. The
liquidity and volume floors ($50k / $75k by default) still apply as a shortlist
filter on top of that, and every candidate still has to clear Rug Check before
it is ever considered for entry. Refreshed every 15 minutes.

It is **Binance.US** (`api.binance.us`), not global Binance — global Binance
geo-blocks US-origin traffic (HTTP 451), which would refuse a droplet hosted
in a US region outright. Binance.US mirrors the same public REST API but with
a smaller listed universe (~150 coins vs. global Binance's thousands).

**Scanning.** Two tiers, both served exclusively by the Jupiter Price API:

- ~15s sweep of the whole eligible universe
- ~1s poll of "tokens of interest" — those already showing a volume spike or
  building momentum, plus every open position

Binance.US is never in this path — it builds the universe shortlist and backs the
candle history (see below), but the live scan is Jupiter Price API only.

**Entry** requires all six to hold at once:

1. volume spike well above the token's *own* rolling average
2. momentum confirmed across several candles, not one noisy print
3. enough pool depth for the intended position size
4. a clean Rug Check result — evaluated *before* the signal, so no scan budget
   is spent reasoning about something that cannot be traded
5. a permitted **market regime**. The efficiency ratio separates a trend from a
   drift; volatility then splits the rest into an orderly range and genuine
   chop. Chop is excluded by default: a volume spike inside a whipsaw looks
   identical to one starting a move, and paying the spread to find out which is
   what produced the earlier fee bill
6. **multi-timeframe agreement**. At the 10-minute default, the 30-minute and
   hourly views must confirm the trend — and only their *closed* bars are read,
   never the one still forming, because a signal that repaints is a signal the
   live bot could never have acted on

Above that sits a **review gate** (below) and hard brakes on trade frequency:
a daily cap, a minimum gap between entries, and a longer cooldown before the
same token can be re-entered.

**Exit** checks three independent conditions every cycle; the first to fire
closes the position:

1. hard stop, placed by the reward:risk window
2. trailing stop, armed only once the move covers round-trip costs (fees +
   slippage), so "breakeven" is never actually a loss
3. signal invalidation — the entry thesis is re-checked continuously and the
   position is closed when it stops holding, which is the main defence against
   capital sitting idle in chop

**Risk.** Volatility-scaled sizing, a correlation check before a second position
(so it is not just riding the first one's move), a portfolio circuit breaker,
congestion awareness, and a gas reserve that position sizing can never dip into.

**Portfolio sizing.** On top of the per-position volatility scaling, `portfolio`
mode solves for the weight that holds *portfolio* volatility at target given
what is already open and how correlated the candidate is with it — a position
riding an open one gets a smaller slice because its marginal contribution to
portfolio risk is larger, not because a heuristic said so. It only ever sizes
down, and stays inside the 45%-per-position and 90%-total caps. The risk budget
itself is scaled by the Monte Carlo **5%-worst-case drawdown**, not the
historical one, which is a single draw from that distribution.

**Entry gate.** Every entry is judged against a short rubric before it executes:
weigh the whole market rather than the single token, favour strong signals over
frequent ones, and prefer capturing a large move with gains locked in over
scalping. Ride-versus-lock-in is decided per trade from the metrics and sets
that position's trailing distance. The rubric is enforced directly by rules —
deterministic, free, instant, auditable, no network call anywhere in the loop.
`entry_gate_min_strength` tunes how demanding it is; turning `entry_gate_enabled`
off approves everything that reaches the gate.

**Drift monitoring.** Live and paper performance is compared continuously
against backtest expectation, on win rate and on expectancy per trade. It is the
earliest warning available: strategies rarely fail loudly, they fail by winning
a little less often than they used to while every individual trade still looks
reasonable.

**Kill switch.** Separate from the circuit breaker. Two-click confirmation, halts
all new entries immediately, never auto-resumes. Open positions are left for
manual closing — Solana swaps are atomic once submitted, so there is no order
book to cancel; the bot simply stops initiating.

---

## Three things the spec asked for that the live APIs no longer support

These were verified against the production APIs on 2026-08-31. The bot is built
to what the APIs actually do; each divergence is called out here rather than
buried.

**1. The Jupiter Price API batches 50 mints per call, not 100.**
`jupiter_price_batch_size` is capped at 50 and the settings page will not accept
more. A 300-token sweep is 6 requests, not 3.

**2. Jupiter's free tier is a rate, not a credit pool, and it is 1 rps.**
`lite-api.jup.ag` — and with it the "25M credits/month" model — was retired on
31 January 2026. Current tiers on `api.jup.ag`: keyless 0.5 rps, free 1 rps,
Developer 10 rps ($25/mo), Launch 50 rps, Pro 150 rps.

The spec's 15s broad sweep plus 1s hot poll needs about **1.4 rps** for a
300-token universe, which the free tier cannot sustain. Rather than silently
falling behind, `Scanner.budget()` computes the requirement on every universe
refresh, surfaces it on the dashboard, and logs a warning. The limiter degrades
the broad sweep first and keeps the hot tier at cadence, so entries and exits
stay responsive while discovery slows. **The $25/mo Developer tier makes the
configured cadence work as specified** — budget for it before going live.

**3. Jupiter's swap flow is now `/order` → sign → `/execute`.**
The old `/quote` + `/swap` pair is gone. Calling `/order` *without* a `taker`
returns a quote with no transaction attached, which is what paper mode and the
pre-trade slippage gate use — cheaper, and it avoids a spurious "missing
associated token account" error for a token the wallet has never held.

### And one thing the spec asked for that does not mean what it used to

**"Liquidity locked" is unmeasurable for most modern Solana tokens.**
Concentrated-liquidity venues — Orca Whirlpool, Meteora DLMM, Raydium CLMM —
have no fungible LP token to burn or lock, so RugCheck reports ~0% locked for
essentially every established token. Applied literally, that filter rejected
**100% of the live tradeable universe** in testing, including ETH and cbBTC.

The filter's *intent* is "the deployer must not be able to pull the rug", and
market count is the honest proxy for that: liquidity spread across hundreds of
independent pools cannot be withdrawn by one actor, whereas a token with a
handful of pools is exactly the case the lock was meant to catch. Live data
bears this out — established tokens show 69–459 markets, a fresh pump.fun token
6.

So the LP-lock threshold is enforced normally, but **waived above
`rugcheck_lp_lock_waiver_markets` pools (default 20)**, with the waiver recorded
in the safety report. Set it to `0` to restore the spec's literal behaviour and
trade nothing. Every other gate still applies — the waiver is not a bypass.

With this in place the gate passes ~3 of the top 14 tokens by volume and rejects
the rest on substantive grounds: retained mint or freeze authority, holder
concentration, danger flags, insufficient age. cbBTC is rejected because
Coinbase retains freeze authority. That is conservative by design.

### A defaults change worth knowing about

`rsi_max_entry` defaults to **78, not ~70**. The momentum rule already requires
several consecutive rising candles, which mechanically drives Wilder RSI into
the mid-70s. A ceiling near 70 double-counts the same condition and leaves
almost no window in which an entry can fire — at 72 the strategy was effectively
inert. 78 still rejects genuinely parabolic moves, which read 82+.

---

## Configuration

Everything is in `config.json`, created on first run from the defaults in
`solbot/config.py`. Nothing is hardcoded at a call site.

Edit it from the dashboard's settings page: changes are bounds-checked, run
through cross-field rules (reward:risk minimum cannot exceed the maximum; sizing
cannot deploy more than 95% of the wallet; the hot scan cannot be slower than
the broad sweep), timestamped into an audit log, and picked up by the worker on
its next cycle via an mtime check. No SSH, no restart.

`trading_mode` is deliberately *not* editable from the dashboard — see
"Going live".

### The parameters most worth tuning

| Setting | Default | Why you would change it |
|---|---|---|
| `min_liquidity_usd` | 50,000 | Raise to trade only deeper, calmer tokens |
| `min_volume_24h_usd` | 75,000 | Raise to cut the universe and the scan cost |
| `volume_spike_multiple` | 2.0 | The single biggest lever on trade frequency |
| `momentum_min_pct` | 0.008 | Raise to demand a stronger move before entering |
| `rr_min` / `rr_max` | 2.0 / 4.0 | The reward:risk window; scaled per trade inside it |
| `trailing_activate_r` | 1.0 | How far in profit before the trail arms |
| `max_slippage_pct` | 0.5 | Trades whose quote exceeds this are skipped entirely |
| `circuit_daily_drawdown_pct` | 0.09 | Daily loss that halts trading until reset |
| `broad_scan_seconds` | 15 | Raise if the dashboard says the cadence is unsustainable |
| `max_trades_per_day` | 8 | The hard brake on overtrading; the other levers are soft |
| `regime_allowed` | 3 | Bitmask: 1 trending, 2 ranging, 4 choppy. Default excludes chop |
| `confluence_required` | 1 | How many higher timeframes must agree before an entry |
| `stop_atr_mult` | 1.2 | Stop distance in ATRs — was hardcoded, now tunable |
| `sizing_mode` | portfolio | `flat` restores per-position volatility scaling alone |
| `drawdown_tolerance` | 0.25 | The 5%-worst-case drawdown you will tolerate; sizing shrinks to fit |
| `shadow_min_days` | 15 | Clean days in shadow before auto-promotion is even considered |
| `entry_gate_min_strength` | 0.0 | Raise to demand a stronger rubric score before an entry is approved |
| `max_total_deployed_pct` | 0.90 | Total wallet exposure cap; there is no cap on *how many* positions make it up |

---

## Backtesting

```bash
python manage.py pull --months 12     # one-time historical fetch from Binance.US
```

```bash
python manage.py backtest --days 90 --verbose
```

Runs the *identical* entry and exit functions the live engine uses, against
locally stored candles, entirely offline. If the backtest and the live bot could
drift apart, the backtest would stop being evidence about the live rules.

**Survivorship bias is handled at the basket level.** The basket comes from
`universe_history` — a point-in-time record of what actually passed the filters
on each historical day — and deliberately keeps tokens that later went to zero.
A basket built from what is tradeable *today* silently excludes every token that
rugged or got delisted, and makes results look considerably better than reality.

That table only accumulates while the bot runs. On a fresh install the backtest
falls back to today's universe and says so in the log; treat those first results
as biased until real snapshots build up.

The whole timeline across every token is walked in chronological order so the
total-deployed cap is respected *across* tokens — there is no cap on how many
positions can be open at once, only on how much of the wallet they can add up
to. Simulating each token in isolation would let the backtest hold more capital
than the live bot ever could, which is its own flattering distortion.

Fees and slippage are charged on both sides. A simulation that ignores costs is
the main way a strategy looks profitable on paper and is not.

A backtest also runs automatically once a day, and the dashboard shows its
metrics next to live and paper performance, so drift between what the rules are
expected to do and what they are actually doing shows up passively.

Use it to sanity-check thresholds — not to hunt for a curve-fit historical
return. Hunting for good parameters is the walk-forward optimizer's job, and it
runs on its own schedule for a reason.

After each daily backtest a **review** proposes at most three parameter
adjustments, backtests the proposed set over the same window, and shadows it
only if it measurably improves on the current one — better return *and* not
materially deeper drawdown. Both sets of numbers are logged either way. Nothing
a review proposes is ever applied to live.

---

## The walk-forward optimizer

The daily backtest tells you whether the current rules are working. The
walk-forward optimizer is what finds better ones — it runs on the droplet
itself every day, and on a rented RunPod GPU once a month for a full sweep.
There is nothing to install or run by hand; the walk-forward tab has manual
trigger buttons if you want either run on demand instead of waiting for its
schedule.

Parameters are tuned on an in-sample window, tested on the next window the
search has never seen, and the whole thing rolls forward. A set is accepted only
if every window that counted produced at least 30 trades, at least 70% of them
were profitable out of sample, and the returns do not swing more than they
average. It is then resampled thousands of times against *observed* execution
costs and replayed through the worst stretches in the history.

An accepted bundle lands straight in SQLite's shadow-instance tables — checked
against the same bounds the settings page enforces, never just trusted because
it came from a run. Promotion to live is automatic after 15 clean days and a
decisive margin — see below.

Everything about it, including how the monthly RunPod worker is provisioned and
torn down, is in [docs/OPTIMIZER.md](docs/OPTIMIZER.md).

---

## Paper, live, and shadow

Three instances, all running the identical rules on identical data; only the
executor differs.

- **paper** — the default, and the required starting mode. Simulates fills with
  realistic fees and slippage, quoting Jupiter for the real cost of the size
  without submitting anything.
- **live** — real swaps. When live, a permanent parallel paper instance keeps
  running alongside so live performance can be compared against expected
  performance on an ongoing basis.
- **shadow** — a candidate rule set running in paper next to the live bot. Set
  overrides by hand from the settings page, or let a daily/monthly walk-forward
  run fill them in.

### Automatic promotion

A shadow set is promoted to live with no manual approval step, but only once
**all** of these hold:

| gate | default | why |
| --- | --- | --- |
| clean days in shadow | 15 | short runs measure luck |
| closed shadow trades | 30 | the same floor a walk-forward window has to clear |
| expectancy margin over live | +25% | "decisive, not marginal" |
| drawdown against live | ≤ 1.15× | extra return bought with extra risk is not free |
| bootstrap confidence | 90% | resampling both trade populations, shadow has to keep winning |
| drift status | not `drifting` | 15 days that stopped meeting expectation are not 15 clean days |

The bootstrap is the one that matters. Comparing two averages says which is
bigger; it does not say whether the difference would survive a different run of
the same two strategies. Every decision is recorded on the walk-forward tab,
including the refusals.

Auto-promotion can be switched off entirely (`auto_promote_enabled`).

The dashboard labels all three distinctly, and the trade journal tags every row
with its instance.

---

## Crash recovery

The spec's hardest requirement, and the one with the most test coverage.

Position state is written to SQLite on *every* change — opening, every trailing
stop advance, every invalidation tick, closing. Nothing important lives only in
memory.

On startup, before the engine takes any new action:

1. open positions are reloaded from SQLite
2. current prices are re-fetched and stop levels recomputed, so a position that
   moved while the process was down is managed against reality
3. when live, state is reconciled against actual on-chain balances — and if the
   database and the wallet disagree, the bot **does not guess**. It halts,
   raises an alert, and waits for you to review and clear it from the dashboard.

The systemd unit restarts on failure, which is safe because the kill switch, the
circuit breaker and the reconciliation halt are all persisted and re-announced
on boot. A restart can never be used, even accidentally, to bypass a triggered
halt. The worker also holds an exclusive lock — a second instance refuses to
start rather than doubling every position.

---

## Dashboard

Dark mode by default, with a light toggle.

- per-position candle charts with entry and exit markers and live stop/trail
  levels drawn on; two open positions render as two clearly labelled charts side
  by side, never merged
- account view: balance, open positions with live unrealised P&L, closed trades
- performance at a glance: win rate, average win vs. average loss, profit
  factor, total fees, max drawdown
- equity curve for every instance
- plain-language live event feed — signals, rejections, breaker trips, restarts,
  not just entries and exits
- start/stop, manual position close, and the two-click kill switch
- settings page with bounds-checked live editing and a change log
- API key rotation with a validation spinner that test-pings the new key and
  says explicitly which key is now in effect — a failed rotation neither
  silently keeps the old key nor silently discards the new one
- progress bars for the historical pull and the daily incremental runs

### Walk-forward tab

Mirrors the daily on-droplet run and the monthly RunPod retest, updated live
over the optimizer endpoints, with buttons to trigger either one on demand:

- the **plain-language run feed** — what the search is discovering as it goes,
  not a percentage bar. "Window 7 out-of-sample: -3.1% from 33 trades, did not
  hold up" is the thing worth knowing four hours in
- a completion notice in the event feed when a run finishes, saying whether it
  produced a promotable set and why not if it did not
- **walk-forward efficiency** as the headline robustness number, alongside how
  many windows were profitable and how far the returns swung
- **OVERFIT** and **FRAGILE** flags, with the failing stress windows named
- per-window pass/fail, including windows that did not reach the trade floor and
  therefore count as neither
- the **Monte Carlo drawdown distribution** as a chart, with the median and the
  5%-worst-case marked, and a second chart plotting the equity curves of the ten
  worst simulated runs directly, not just their depth as a number — saying
  whether execution costs were measured or assumed
- the **drift panel**: live and paper against backtest expectation over time
- the parameter pipeline: every bundle received, its status, and every promotion
  decision with its reasoning
- a **stored WF/MC runs** panel, browsable by date, that ages out past its own
  retention window (`wfmc_result_retention_days`) — unlike raw candle history,
  kept indefinitely

Charts are drawn by a small local canvas renderer (`static/charts.js`) rather
than a CDN library: the dashboard ships a strict CSP that forbids third-party
scripts, and all rendering happens in the browser so the droplet's single vCPU
never touches chart work. To swap in Lightweight-Charts, vendor the file into
`static/` and reimplement the two render functions.

### Security (required before live trading)

- login on every route except the login screen itself, enforced by a
  `before_request` hook — a route added later is protected by default rather
  than by remembering a decorator
- argon2id password hashing
- app-based TOTP via `pyotp` (Google Authenticator / Authy), QR enrolment,
  ten single-use backup codes. No SMS, no third party, no phone number.
- session cookies `Secure`, `HttpOnly`, `SameSite=Strict`, with a sliding timeout
- login attempts rate-limited by IP, in the app and again at nginx
- every login, failed attempt, kill-switch trigger and settings change is logged
- HTTPS via DuckDNS + Certbot — free, no domain purchase. Let's Encrypt will not
  issue for a bare IP, which is why the free subdomain is needed.

---

## Wallet security

The bot needs its own **freshly generated, dedicated** keypair:

```bash
python manage.py new-wallet
```

Never reuse a personal wallet. Fund it only with capital you are fully willing to
lose. Treat it as an isolated hot wallet.

The private key is loaded from the environment at runtime, lives only inside the
executor, and never reaches the web process, any log, or any file that touches
version control. That reduces exposure; it does not eliminate it. A server
running a live trading loop and a web dashboard with a hot wallet key carries
non-zero risk, and there is no fully airtight version of this.

---

## Going live

```bash
python manage.py go-live
```

Refuses unless the wallet key is set, the Flask secret is set, a dashboard user
exists *with confirmed TOTP*, at least 20 paper trades have been recorded, at
least one backtest has been run, and a Jupiter API key is set. Then it makes you
type `go live` in full.

It cannot check that HTTPS actually works or that the dashboard is not reachable
on the bare IP. Those are on you, and they matter — that dashboard can close
positions and rotate keys.

---

## Exports

```bash
python manage.py export --instance live --out trades.csv
```

```bash
python manage.py tax --year 2026 --out tax-2026.csv
```

The tax summary covers **live trades only** — paper and shadow fills are
excluded because they are not real taxable events. It totals realized gains and
losses per token and for the period, split short/long term, structured to hand
to a preparer. Both are also downloadable from the dashboard.

This is a summary of your own trade record, not tax advice.

---

## Tests

```bash
python -m pytest
```

321 tests, no network access — every API client (Jupiter, Binance, RugCheck,
Solana RPC, RunPod) is faked. A test that needs an API key is a test that does
not run, and there is no live model call anywhere in this bot to stub out in
the first place — the entry gate is pure rules, so it is exercised the same way
any other deterministic function is.

Coverage concentrates on what loses money when it breaks: entry/exit rules,
position sizing, the circuit breaker, crash recovery, on-chain reconciliation,
the safety gate's fail-closed behaviour, rate-limit headroom, and the
survivorship-bias handling in the backtest.

Four groups are worth calling out:

- **`test_vector_parity.py`** asserts the optimizer computes exactly what the
  live bot computes — every indicator bar for bar against pandas, and the
  vectorized engine trade for trade against the shipped backtester on identical
  data. The whole value of the optimizer rests on that claim.
- **`test_review_layer.py`** pins what the entry gate approves and rejects
  under the rubric, and that the gate can be switched off entirely
  (`entry_gate_enabled`) without touching the rules engine.
- **`test_candlestore.py`** pins the Parquet append path — incremental writes
  merge and dedupe against what is already on disk rather than duplicating it,
  and never touch the still-forming bar.
- **`test_runpod.py`** pins the monthly-retest orchestration against a fake
  transport: volume and pod lifecycle, and — the one that actually matters —
  that teardown verification catches a worker that did not self-terminate and
  force-stops it rather than leaving a GPU billing in the background. No test
  here ever makes a live RunPod call.

**`test_optimizer_api.py`** covers the parameter-bundle ingest endpoint that
replaced the old bulk-download API — including that a logged-in dashboard
session grants no special access to it, so a RunPod worker's bearer token and
an operator's session stay two separate privilege levels.

---

## Layout

```
solbot/
  config.py        settings, bounds validation, hot reload
  db.py            SQLite schema and helpers (trades/positions/settings/audit/WF-MC)
  ratelimit.py     token bucket with a reserved lane for high-priority calls
  clients/         jupiter, binance, rugcheck, solana rpc
  candlestore.py   Parquet candle store - append, read, coverage, disk usage
  indicators.py    EMA, RSI, ATR, volume ratio, resampling
  strategy.py      entry and exit rules (pure functions)
  risk.py          sizing, correlation, circuit breaker, kill switch
  safety.py        Rug Check gate with cooldown-based re-testing
  universe.py      Binance top-N -> Jupiter routing + point-in-time snapshots
  scanner.py       two-tier polling and the cadence budget
  datastore.py     Binance backfill (bulk + daily incremental), live tick roll-up
  execution.py     paper and live executors
  portfolio.py     positions, balances, journal, performance
  recovery.py      startup recovery and on-chain reconciliation
  engine.py        the trading loop
  backtest.py      offline simulation
  regime + confluence live in indicators.py; the gates are in strategy.py
  drift.py         live vs backtest drift monitoring
  review.py        the rules-based entry gate
  daily_review.py  post-backtest parameter proposals (shadow only)
  wfmc.py          daily on-droplet run, monthly RunPod orchestration
  runpod.py        RunPod volume/pod lifecycle + teardown verification
  paramsync.py     bundle validation, shadow install, auto-promotion
  exports.py       CSV and tax exports
  web/             Flask dashboard, auth, JSON API, templates
                   optimizer.py = parameter-bundle ingest + run-feed ingest

solopt/            the vectorized engine - imports no solbot, invoked from two
                   places: solbot/wfmc.py (daily, in-process) and a RunPod
                   worker (monthly, via cli.py)
  arrays.py        CuPy/NumPy backend and the utilization throttle
  schema.py        asset-class description and cost model (crypto/equity/fx)
  dataset.py       bundle loading, roll-up, point-in-time eligibility
  frames.py        compaction into per-symbol bar sequences
  indicators.py    batched indicator math, parity-tested against solbot
  params.py        search space, plateau and cap stopping rules
  engine.py        the vectorized backtest engine
  walkforward.py   rolling windows, thresholds, per-coin params, resume
  montecarlo.py    resampling the out-of-sample trades, worst-path tracking
  stress.py        crash and flash-crash discovery and replay
  promotion.py     parameter bundle assembly and its gates
  pipeline.py      shared run_pipeline() - the one orchestration both call sites use
  report.py        pushing feed/run/bundle back to the droplet over HTTP
  cli.py           python -m solopt {run,status,feed} - the RunPod worker's entry point

deploy/            systemd units, nginx config, deployment guide
docs/OPTIMIZER.md  how the optimizer works and how to run it
tests/
```

---

## Risk

This trades real money on volatile assets. It is designed to survive rather than
to maximise return, and the defaults are conservative for that reason — but no
amount of engineering makes an automated trading bot safe. Start in paper mode,
leave it there until the numbers mean something, and fund the wallet only with
what you can afford to lose entirely.
