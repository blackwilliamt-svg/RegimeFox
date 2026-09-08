#!/usr/bin/env python3
"""Command-line administration.

    python manage.py init-db
    python manage.py new-wallet
    python manage.py create-user <username>
    python manage.py backup-codes <username>
    python manage.py universe
    python manage.py pull --months 12
    python manage.py backtest --days 90
    python manage.py export --instance live --out trades.csv
    python manage.py tax --year 2026
    python manage.py status
    python manage.py check-apis
    python manage.py check-deploy     (warns if config.json/db are missing)
    python manage.py benchmark-runpod (compares GPU tiers, launches real pods)
    python manage.py go-live          (guarded checklist)
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The Windows console defaults to cp1252; without this any non-ASCII output
# raises UnicodeEncodeError instead of printing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from solbot import db, exports, risk
from solbot.backtest import run_and_store
from solbot.clients import build_clients
from solbot.config import get_config
from solbot.datastore import DataStore
from solbot.universe import UniverseBuilder


def cmd_init_db(_args: argparse.Namespace) -> int:
    db.init_db()
    print(f"database ready at {db.db_path()}")
    cfg = get_config()
    if cfg.ensure_on_disk():
        print(f"config.json created at {cfg.path} (defaults)")
    return 0


def cmd_check_deploy(_args: argparse.Namespace) -> int:
    """Warn about the local conditions that make a fresh systemd install fail.

    Currently just the config.json-must-exist-before-bind-mount bug: both
    solbot-worker.service and solbot-web.service declare
    `ReadWritePaths=... config.json` under ProtectSystem=strict, and systemd
    refuses to start (226/NAMESPACE) if that path doesn't exist yet.
    """
    problems: list[str] = []
    cfg = get_config()
    if not cfg.path.exists():
        problems.append(
            f"{cfg.path} does not exist yet - systemd's ReadWritePaths= for it "
            "will fail to bind-mount (226/NAMESPACE) until it does. "
            "Run: python manage.py init-db"
        )
    if not db.db_path().exists():
        problems.append(
            f"{db.db_path()} does not exist yet. Run: python manage.py init-db"
        )
    if problems:
        for p in problems:
            print(f"[warn] {p}")
        return 1
    print("[ok]   config.json and the database both exist")
    return 0


def cmd_new_wallet(_args: argparse.Namespace) -> int:
    """Generate a fresh, dedicated hot wallet keypair."""
    try:
        from solders.keypair import Keypair
        import base58
    except ImportError:
        print("install the live-trading extras first:  pip install solders base58")
        return 1

    kp = Keypair()
    secret = base58.b58encode(bytes(kp)).decode("ascii")
    print("\n  A NEW wallet has been generated. It is not saved anywhere.\n")
    print(f"  Public key : {kp.pubkey()}")
    print(f"  Private key: {secret}\n")
    print("  Put the private key in .env as SOLANA_PRIVATE_KEY, then:")
    print("    - fund it ONLY with capital you are willing to lose")
    print("    - never reuse a personal wallet (Phantom etc.) for this")
    print("    - keep some SOL above the gas reserve so fees can always be paid")
    print("    - back the key up somewhere offline; this is the only time it is shown\n")
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    from solbot.web.auth import create_user, provisioning_uri

    db.init_db()
    password = getpass.getpass("Password (12+ chars): ")
    if password != getpass.getpass("Confirm: "):
        print("passwords do not match")
        return 1
    try:
        secret, codes = create_user(args.username, password)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    print(f"\n  User {args.username!r} created.\n")
    print(f"  TOTP secret : {secret}")
    print(f"  Enrolment   : {provisioning_uri(args.username, secret)}\n")
    print("  Backup codes (each works once, save them now):")
    for c in codes:
        print(f"    {c}")
    print("\n  Add the secret to Google Authenticator or Authy, then sign in.")
    print("  The QR code is also shown at /setup on a fresh install.\n")
    return 0


def cmd_backup_codes(args: argparse.Namespace) -> int:
    from solbot.web.auth import get_user, regenerate_backup_codes

    db.init_db()
    if get_user(args.username) is None:
        print(f"no such user: {args.username}")
        return 1
    codes = regenerate_backup_codes(args.username)
    print("New backup codes (the old ones are now invalid):")
    for c in codes:
        print(f"  {c}")
    return 0


def cmd_universe(_args: argparse.Namespace) -> int:
    cfg = get_config()
    db.init_db()
    clients = build_clients(cfg)
    builder = UniverseBuilder(clients.binance, clients.jupiter, cfg.as_dict())
    stats = builder.refresh()
    print(
        f"universe: {stats.passed} tradeable of {stats.considered} considered\n"
        f"  rejected: {stats.rejected_liquidity} on liquidity, "
        f"{stats.rejected_volume} on volume, {stats.rejected_excluded} excluded"
    )
    for t in sorted(builder.tokens.values(), key=lambda x: -x.volume_24h)[:15]:
        print(
            f"  {t.symbol:<12} liq ${t.liquidity:>12,.0f}  "
            f"vol24h ${t.volume_24h:>13,.0f}  holders {t.holder_count:>7,}"
        )
    clients.close()
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = get_config()
    db.init_db()
    clients = build_clients(cfg)
    store = DataStore(clients.binance, cfg.as_dict())
    conn = db.connect()

    builder = UniverseBuilder(clients.binance, clients.jupiter, cfg.as_dict())
    if not builder.load_persisted(conn):
        builder.refresh(conn=conn)
    pairs = store.pair_map(conn)
    if args.limit:
        pairs = dict(list(pairs.items())[: args.limit])

    estimate = store.estimate_pull(len(pairs), args.months)
    print(json.dumps(estimate, indent=2))

    def progress(done: int, total: int, pair: str) -> None:
        pct = 100.0 * done / total if total else 0
        print(f"\r  {done}/{total} ({pct:5.1f}%) {pair}...", end="", flush=True)

    report = store.bulk_backfill(pairs, months=args.months, conn=conn, progress=progress)
    print()
    print(json.dumps(report.as_dict(), indent=2))
    clients.close()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = get_config()
    db.init_db()
    clients = build_clients(cfg)
    store = DataStore(clients.binance, cfg.as_dict())

    result = run_and_store(store, cfg.as_dict(), days=args.days)
    summary = result.summary()
    print(json.dumps(summary, indent=2))

    if result.trades and args.verbose:
        print("\nfirst 20 trades:")
        for t in result.trades[:20]:
            print(
                f"  {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t.entry_ts))} "
                f"{t.symbol:<8} {t.pnl_usd:+9.2f}  {t.exit_reason}"
            )
    clients.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    db.init_db()
    rows = exports.fetch_trades(instance=args.instance)
    body = exports.trades_csv(rows)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"wrote {len(rows)} trades to {args.out}")
    else:
        print(body)
    return 0


def cmd_tax(args: argparse.Namespace) -> int:
    db.init_db()
    summary = exports.tax_summary(year=args.year)
    body = exports.tax_csv(summary)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"wrote tax summary to {args.out}")
    else:
        print(body)
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    from solbot.portfolio import Portfolio

    cfg = get_config()
    db.init_db()
    conn = db.connect()

    hb = db.kv_get("worker_heartbeat", {}) or {}
    age = db.now() - int(hb.get("ts", 0)) if hb.get("ts") else None

    print(f"mode              : {cfg['trading_mode']}")
    print(
        f"worker            : "
        + (f"alive, cycle {hb.get('cycle')}, {age}s ago" if age is not None and age < 60
           else f"NOT RESPONDING (last seen {age}s ago)" if age is not None else "never started")
    )
    print(f"trading enabled   : {bool(db.kv_get('engine_run', True))}")
    print(f"kill switch       : {risk.kill_switch_state(conn)}")
    print(f"circuit breaker   : {risk.circuit_state(conn)}")

    row = conn.execute("SELECT COUNT(*) AS n FROM universe").fetchone()
    print(f"universe          : {row['n']} tokens")

    clients = build_clients(cfg)
    store = DataStore(clients.binance, cfg.as_dict())
    cov = store.coverage(conn)
    print(
        f"candle coverage   : {cov['tokens']}/{row['n']} tokens have history, "
        f"{cov['candles']:,} candles, {cov['disk']['megabytes']:,.1f} MB on disk"
    )
    clients.close()

    for name in db.INSTANCES:
        p = Portfolio(name, cfg.as_dict())
        perf = p.performance(conn=conn)
        if not perf["trades"] and p.open_count(conn) == 0:
            continue
        print(
            f"\n[{name}] balance ${p.balance(conn):,.2f}  deployed "
            f"${p.deployed_usd(conn):,.2f}  open {p.open_count(conn)}"
        )
        print(
            f"    {perf['trades']} trades, {perf['win_rate'] * 100:.1f}% win rate, "
            f"P&L ${perf['total_pnl']:,.2f}, fees ${perf['total_fees']:,.2f}, "
            f"max DD {perf['max_drawdown'] * 100:.1f}%"
        )
    return 0


def cmd_check_apis(_args: argparse.Namespace) -> int:
    """Verify every API is reachable and every key works."""
    cfg = get_config()
    db.init_db()
    clients = build_clients(cfg)
    ok = True

    checks = [
        ("Jupiter price", lambda: clients.jupiter.ping()),
        ("Jupiter tokens", lambda: bool(clients.jupiter.top_tokens(limit=5))),
        ("RugCheck", lambda: clients.rugcheck.ping()),
        ("Solana RPC", lambda: clients.rpc.ping()),
        ("Binance", lambda: clients.binance.ping()),
    ]

    for name, fn in checks:
        try:
            fn()
            print(f"  [ok]   {name}")
        except Exception as exc:
            ok = False
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")

    print(f"\n  jupiter key: {'set' if cfg.secrets.jupiter_api_key else 'NOT SET (keyless, 0.5 rps)'}")
    print(f"  wallet key : {'set' if cfg.secrets.solana_private_key else 'not set (paper only)'}")
    clients.close()
    return 0 if ok else 1


def cmd_benchmark_runpod(args: argparse.Namespace) -> int:
    """Benchmark 2-3 GPU tiers on a small representative slice (gap-closure
    item 7) and report cost-per-run for each - the dashboard's RunPod
    settings panel has the same button, backed by the same function."""
    from solbot.wfmc import run_benchmark

    cfg = get_config()
    db.init_db()
    clients = build_clients(cfg)
    store = DataStore(clients.binance, cfg.as_dict())
    tiers = args.tiers.split(",") if args.tiers else None

    print("Benchmarking RunPod GPU tiers - this launches real (billed) pods and can take a while.")
    result = run_benchmark(cfg.as_dict(), store, cfg.secrets, tiers=tiers)
    clients.close()

    if not result.get("ran"):
        print(f"  did not run: {result.get('reason')}")
        return 1

    print()
    for r in result["results"]:
        status = "ok" if r["ok"] else f"FAILED: {r['error']}"
        cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "cost unknown"
        per1k = (
            f", ${r['cost_per_1000_combinations']:.4f}/1000 combos"
            if r["cost_per_1000_combinations"] is not None else ""
        )
        teardown = "clean" if r["teardown_clean"] else "NOT CLEAN" if r["teardown_clean"] is False else "unknown"
        print(f"  [{status}] {r['gpu_type']}: {r['elapsed_seconds']:.0f}s, {cost}{per1k}, teardown {teardown}")

    if result.get("recommendation"):
        print(f"\n  Recommendation: {result['recommendation']}")
        print(
            "  This does NOT change runpod_gpu_type automatically - update it "
            "from the settings page if you agree with it."
        )
    return 0


def cmd_go_live(args: argparse.Namespace) -> int:
    """Switch to live trading, but only after the checklist actually passes."""
    from solbot.web.auth import user_count

    cfg = get_config()
    db.init_db()
    conn = db.connect()
    problems: list[str] = []

    if not cfg.secrets.solana_private_key:
        problems.append("SOLANA_PRIVATE_KEY is not set (run: python manage.py new-wallet)")
    if not cfg.secrets.flask_secret_key:
        problems.append("FLASK_SECRET_KEY is not set")
    if user_count(conn) == 0:
        problems.append("no dashboard user exists (run: python manage.py create-user <name>)")

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM users WHERE totp_confirmed = 1"
    ).fetchone()
    if int(row["n"]) == 0:
        problems.append("no dashboard user has confirmed TOTP two-factor enrolment")

    paper = conn.execute(
        "SELECT COUNT(*) AS n FROM trades WHERE instance = 'paper'"
    ).fetchone()
    if int(paper["n"]) < 20:
        problems.append(
            f"only {paper['n']} paper trades recorded; run paper mode until there is a "
            "real sample before risking funds"
        )

    backtest = conn.execute("SELECT COUNT(*) AS n FROM backtest_runs").fetchone()
    if int(backtest["n"]) == 0:
        problems.append("no backtest has ever been run")

    if not cfg.secrets.jupiter_api_key:
        problems.append(
            "JUPITER_API_KEY is not set; keyless access is 0.5 rps and cannot sustain "
            "the scan cadence for live trading"
        )

    if problems:
        print("Not ready for live trading:\n")
        for p in problems:
            print(f"  [FAIL] {p}")
        print("\nAlso confirm manually, because this script cannot check it:")
        print("  - HTTPS is working (DuckDNS + Certbot), not plain HTTP")
        print("  - the dashboard is not reachable on the droplet's bare IP")
        print("  - the wallet holds only money you are willing to lose")
        if not args.force:
            return 1
        print("\n--force given; continuing anyway.")

    print("\nThis switches TRADING MODE TO LIVE. Real funds will be traded.")
    if input("Type 'go live' to confirm: ").strip().lower() != "go live":
        print("aborted")
        return 1

    cfg.set_trading_mode("live")
    db.log_event(
        "Trading mode switched to LIVE from the command line.",
        level="alert",
        category="system",
        conn=conn,
    )
    print("\nmode is now 'live'. Restart the worker:  sudo systemctl restart solbot-worker")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Solana TA bot administration")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the database schema").set_defaults(fn=cmd_init_db)
    sub.add_parser("new-wallet", help="generate a dedicated hot wallet").set_defaults(
        fn=cmd_new_wallet
    )

    p = sub.add_parser("create-user", help="create a dashboard user with TOTP")
    p.add_argument("username")
    p.set_defaults(fn=cmd_create_user)

    p = sub.add_parser("backup-codes", help="regenerate a user's backup codes")
    p.add_argument("username")
    p.set_defaults(fn=cmd_backup_codes)

    sub.add_parser("universe", help="refresh and print the tradeable universe").set_defaults(
        fn=cmd_universe
    )

    p = sub.add_parser("pull", help="backfill historical candles from Binance")
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="cap the number of tokens")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("backtest", help="run a backtest against stored candles")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("export", help="export the trade journal as CSV")
    p.add_argument("--instance", default=None, choices=[*db.INSTANCES, None])
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("tax", help="export the live-trades-only tax summary")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_tax)

    sub.add_parser("status", help="print engine and portfolio status").set_defaults(
        fn=cmd_status
    )
    sub.add_parser("check-apis", help="verify API reachability and keys").set_defaults(
        fn=cmd_check_apis
    )
    sub.add_parser(
        "check-deploy",
        help="warn if config.json/the database are missing (systemd bind-mount bug)",
    ).set_defaults(fn=cmd_check_deploy)

    p = sub.add_parser(
        "benchmark-runpod",
        help="benchmark 2-3 GPU tiers and report cost-per-run (launches real, billed pods)",
    )
    p.add_argument(
        "--tiers", default=None,
        help="comma-separated GPU tier names to test (default: a built-in shortlist)",
    )
    p.set_defaults(fn=cmd_benchmark_runpod)

    p = sub.add_parser("go-live", help="switch to live trading after the checklist")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_go_live)

    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
