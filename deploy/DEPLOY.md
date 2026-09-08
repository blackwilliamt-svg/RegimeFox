# Deploying to the DigitalOcean droplet

Target: 1 vCPU / 2 GB (Basic Intel). This bot runs in its own directory, its own
virtualenv and its own systemd units, entirely separate from the freight lead-gen
app already on the box — nothing is shared except nginx and the OS.

Two processes:

| Unit | Role |
|---|---|
| `solbot-worker` | the trading loop. The only process that trades. |
| `solbot-web` | gunicorn serving the dashboard on `127.0.0.1:8090`. |

They share **only** the SQLite database. The dashboard never opens or closes a
position; it writes a row to the `commands` table that the worker drains. That
keeps a single writer on the trading path.

---

## 1. System packages

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip nginx git
```

## 2. Create a dedicated user

Do not run this as root. It holds a hot wallet key.

```bash
sudo useradd --system --create-home --home-dir /opt/solana-ta-bot --shell /usr/sbin/nologin solbot
```

## 3. Install the code

```bash
sudo -u solbot git clone <your-repo-url> /opt/solana-ta-bot
cd /opt/solana-ta-bot
sudo -u solbot python3 -m venv .venv
sudo -u solbot .venv/bin/pip install --upgrade pip
sudo -u solbot .venv/bin/pip install -r requirements.txt
```

## 4. Configure secrets

```bash
sudo -u solbot cp .env.example .env
sudo -u solbot chmod 600 .env
sudo -u solbot nano .env
```

Generate the two app secrets:

```bash
python3 -c "import secrets;print('FLASK_SECRET_KEY=' + secrets.token_hex(32))"
```

```bash
python3 -c "import secrets;print('SECRET_ENCRYPTION_KEY=' + secrets.token_hex(32))"
```

API keys:

- **Jupiter** — <https://portal.jup.ag>, Developer Platform → create key.
  The free tier is **1 request/second**; the two-tier scan cadence really wants
  the $25/month Developer tier at 10 rps. See "Rate limits" below.

Binance needs no key — universe ranking and candle history (klines + monthly
archives) are all keyless public endpoints.

Leave `SOLANA_PRIVATE_KEY` blank for now. Paper mode does not need it. Leave
`RUNPOD_API_KEY` / `RUNPOD_S3_ACCESS_KEY` / `RUNPOD_S3_SECRET_KEY` blank too —
they are only needed for the monthly RunPod retest (`wfmc_monthly_enabled`,
off by default); see the "Walk-forward / Monte Carlo" section below.

## 5. Initialise and create your dashboard login

```bash
sudo -u solbot .venv/bin/python manage.py init-db
```

This also creates `config.json` (with default values) if it doesn't already
exist. That matters for step 6: both systemd units declare
`ReadWritePaths=... config.json` under `ProtectSystem=strict`, and systemd can
only bind-mount a path that already exists — skipping `init-db` before
`systemctl enable --now` fails with status 226/NAMESPACE. Run
`manage.py check-deploy` at any point to confirm both `config.json` and the
database are in place before you get to step 6.

```bash
sudo -u solbot .venv/bin/python manage.py create-user yourname
```

That prints a TOTP secret and ten single-use backup codes. Add the secret to
Google Authenticator or Authy and **save the backup codes somewhere offline** —
they are shown once.

Verify everything is reachable:

```bash
sudo -u solbot .venv/bin/python manage.py check-apis
```

## 6. systemd units

```bash
sudo cp deploy/solbot-worker.service deploy/solbot-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now solbot-worker solbot-web
```

```bash
sudo systemctl status solbot-worker solbot-web
```

## 7. HTTPS — free, no domain purchase

Let's Encrypt will not issue a certificate for a bare IP address, so you need a
hostname. DuckDNS gives you one free.

1. Sign in at <https://duckdns.org>, create a subdomain, point it at the
   droplet's IP.
2. Install the nginx site:

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/solbot
sudo sed -i 's/YOUR-SUBDOMAIN.duckdns.org/your-actual-subdomain.duckdns.org/g' /etc/nginx/sites-available/solbot
sudo ln -sf /etc/nginx/sites-available/solbot /etc/nginx/sites-enabled/solbot
sudo nginx -t && sudo systemctl reload nginx
```

3. Issue the certificate (auto-renewing):

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-actual-subdomain.duckdns.org
```

4. Confirm the renewal timer is armed:

```bash
sudo systemctl list-timers | grep certbot
```

**Do not skip this.** TOTP protects the login, but not the session cookie
afterwards — over plain HTTP a session can be hijacked in transit, and that
session can trigger the kill switch and close positions.

## 8. Lock down the firewall

```bash
sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw --force enable
```

Port 8090 is never exposed — gunicorn binds to localhost only. Verify:

```bash
sudo ss -tlnp | grep 8090
```

You should see `127.0.0.1:8090`, never `0.0.0.0:8090`.

---

## Rate limits — read this before tuning the scan cadence

Jupiter's pricing changed in 2026; `lite-api.jup.ag` was retired on 31 January.
Current tiers on `api.jup.ag`:

| Tier | Rate | Cost |
|---|---|---|
| Keyless | 0.5 rps | free, no signup |
| Free | 1 rps | free |
| Developer | 10 rps | $25/mo |
| Launch | 50 rps | $100/mo |
| Pro | 150 rps | $500/mo |

The Price API accepts **50 mints per call**. One sweep of a 300-token universe
is therefore 6 requests. The spec's 15s broad sweep plus a 1s hot poll needs
roughly **1.4 rps**, which the free tier cannot sustain.

The bot does not pretend otherwise. `Scanner.budget()` computes the requirement
every universe refresh, writes it to the dashboard, and logs a warning when the
configured cadence outruns the key. Your options, in order of preference:

1. Move to the Developer tier ($25/mo) and the configured cadence just works.
2. Raise `broad_scan_seconds`, or lower `universe_max_tokens`, until the
   dashboard reports the cadence as sustainable.
3. Leave it — the limiter degrades the *broad* sweep first and protects the hot
   tier, so entries and exits stay responsive while discovery slows down.

**Binance.US** (not global Binance - see below) is keyless and rate-limited
generously enough (`binance_rps`, 5 rps default) that neither the universe
refresh nor the daily incremental candle pull needs a budget check. The
one-time bulk backfill (a trailing year of 1-minute candles per coin, paged
through the same klines endpoint month by month) is still a manual,
progress-reported button rather than something that runs on boot, simply
because it is a heavy one-time load you should kick off deliberately.

**Why Binance.US specifically:** global Binance (`api.binance.com`) geo-blocks
US-origin traffic outright (HTTP 451). A droplet in a US region gets refused
by it entirely, so this bot talks to `api.binance.us` instead - same REST API
shape, keyless, but a smaller listed universe (~150 coins vs. thousands) and
no documented equivalent of global Binance's bulk monthly-archive download,
hence the REST paging above. If your droplet is hosted **outside** the US,
global Binance would also work and has the larger coin selection - but there
is no config flag to switch back; it would mean reverting
`solbot/clients/binance.py`'s `base_url`.

---

## Day-to-day

```bash
sudo journalctl -u solbot-worker -f
```

```bash
sudo -u solbot /opt/solana-ta-bot/.venv/bin/python /opt/solana-ta-bot/manage.py status
```

Updating:

```bash
cd /opt/solana-ta-bot && sudo -u solbot git pull && sudo -u solbot .venv/bin/pip install -r requirements.txt && sudo systemctl restart solbot-worker solbot-web
```

### Walk-forward / Monte Carlo (spec 5)

The daily re-score runs inside `solbot-worker` itself, on this droplet's own
CPU — nothing to set up beyond the defaults (`wfmc_daily_enabled`, on by
default). There is no PC and no separate hand-off repository anywhere in this
picture; an accepted parameter set goes straight into SQLite.

The monthly full-space retest (`wfmc_monthly_enabled`, **off** by default)
rents a GPU worker from RunPod for the job and needs its own credentials:

1. **RunPod API key + S3 credentials.** Put `RUNPOD_API_KEY`,
   `RUNPOD_S3_ACCESS_KEY` and `RUNPOD_S3_SECRET_KEY` in `.env`.
2. **A worker token for this droplet.** Dashboard → Settings → *RunPod worker
   token* → *Generate a new token*. It is shown exactly once, and grants write
   access to the run feed and bundle submission only — it cannot halt trading,
   close a position, change a setting, or read an API key.
3. **This droplet's own public URL.** Set `runpod_callback_url` on the settings
   page so a RunPod worker knows where to report its feed and finished bundle
   back to.
4. Turn on `wfmc_monthly_enabled` once the above is in place.

The droplet ships the worker its candle data, waits for it to report back,
then terminates it and **verifies the teardown** — logged to the event feed
either way, so a worker that failed to self-stop never keeps billing silently.

Full detail, including the daily/monthly architecture, in
[../docs/OPTIMIZER.md](../docs/OPTIMIZER.md).

### Backups

The database holds your entire trade history, settings, audit log, and WF/MC
results — back it up. Candle history (Parquet, under `data/candles/`) is
deliberately **not** backed up off the droplet — it is kept indefinitely and
rebuilt cheaply from Binance if ever lost, so it is not worth the storage cost
of a second copy.

```bash
sudo -u solbot sqlite3 /opt/solana-ta-bot/data/solbot.db ".backup '/opt/solana-ta-bot/data/backup-$(date +%F).db'"
```

Add it to cron. `VACUUM INTO` / `.backup` is safe to run while the worker is
writing; copying the file directly is not.

---

## Going live

Do not flip the switch by editing `config.json` by hand. Use:

```bash
sudo -u solbot /opt/solana-ta-bot/.venv/bin/python /opt/solana-ta-bot/manage.py go-live
```

It refuses unless every one of these holds:

- `SOLANA_PRIVATE_KEY` is set (generate with `manage.py new-wallet` — a fresh,
  dedicated wallet, never a personal Phantom one)
- `FLASK_SECRET_KEY` is set
- a dashboard user exists **and has confirmed TOTP enrolment**
- at least 20 paper trades have been recorded
- at least one backtest has been run
- `JUPITER_API_KEY` is set

It then asks you to type `go live` in full. Restart the worker afterwards.

Things the script cannot check, which are on you:

- HTTPS actually works and the dashboard is not reachable on the bare IP
- the wallet holds only money you are willing to lose
