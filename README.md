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

**Universe.** Built from a liquidity floor and a volume floor ($50k / $75k by
default), not a fixed token count, so it grows and shrinks with what is actually
tradeable. Those defaults deliberately exclude micro-caps, which move too
erratically for a rules-based approach. Refreshed every 15 minutes from Jupiter's
token rankings. On live data that currently yields ~90 tradeable tokens.

**Scanning.** Two tiers, both served exclusively by the Jupiter Price API:

- ~15s sweep of the whole eligible universe
- ~1s poll of "tokens of interest" — those already showing a volume spike or
  building momentum, plus every open position

Birdeye is never in this path. Its free tier is 1 request/second in total and
would be exhausted immediately; it is used only for historical backfill.

**Entry** requires all four to hold at once:

1. volume spike well above the token's *own* rolling average
2. momentum confirmed across several candles, not one noisy print
3. enough pool depth for the intended position size
4. a clean Rug Check result — evaluated *before* the signal, so no scan budget
   is spent reasoning about something that cannot be traded

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

---

## Backtesting

```bash
python manage.py pull --days 90       # one-time historical fetch (budget-checked)
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
concurrent-position cap is respected *across* tokens. Simulating each token in
isolation would let the backtest hold far more positions than the live bot can,
which is its own flattering distortion.

Fees and slippage are charged on both sides. A simulation that ignores costs is
the main way a strategy looks profitable on paper and is not.

A backtest also runs automatically once a day, and the dashboard shows its
metrics next to live and paper performance, so drift between what the rules are
expected to do and what they are actually doing shows up passively.

Use it to sanity-check thresholds — not to hunt for a curve-fit historical
return.

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
  overrides from the settings page. Nothing is ever promoted automatically.

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

153 tests, no network access — every API client is faked. Coverage concentrates
on the things that lose money when they break: entry/exit rules, position
sizing, the circuit breaker, crash recovery, on-chain reconciliation, the safety
gate's fail-closed behaviour, rate-limit headroom, and the survivorship-bias
handling in the backtest.

---

## Layout

```
solbot/
  config.py        settings, bounds validation, hot reload
  db.py            SQLite schema and helpers
  ratelimit.py     token bucket with a reserved lane for high-priority calls
  clients/         jupiter, birdeye, rugcheck, solana rpc
  indicators.py    EMA, RSI, ATR, volume ratio, resampling
  strategy.py      entry and exit rules (pure functions)
  risk.py          sizing, correlation, circuit breaker, kill switch
  safety.py        Rug Check gate with cooldown-based re-testing
  universe.py      universe building + point-in-time snapshots
  scanner.py       two-tier polling and the cadence budget
  datastore.py     candle storage, backfill, retention
  execution.py     paper and live executors
  portfolio.py     positions, balances, journal, performance
  recovery.py      startup recovery and on-chain reconciliation
  engine.py        the trading loop
  backtest.py      offline simulation
  exports.py       CSV and tax exports
  web/             Flask dashboard, auth, JSON API, templates
deploy/            systemd units, nginx config, deployment guide
tests/
```

---

## Risk

This trades real money on volatile assets. It is designed to survive rather than
to maximise return, and the defaults are conservative for that reason — but no
amount of engineering makes an automated trading bot safe. Start in paper mode,
leave it there until the numbers mean something, and fund the wallet only with
what you can afford to lose entirely.
