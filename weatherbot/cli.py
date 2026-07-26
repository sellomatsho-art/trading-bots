"""Command line entry point.

    python -m weatherbot scan        one pass: forecast, price, book paper fills
    python -m weatherbot settle      score past-dated positions against observations
    python -m weatherbot calibrate   learn per-city forecast bias from history
    python -m weatherbot report      current paper PnL
    python -m weatherbot loop        scan + settle on an interval
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

from .calibration import learn_bias
from .cities import CITIES
from .config import Config
from .engine import recalibrate, scan, settle_pending
from .ledger import Ledger
from .markets import MarketDataError

BANNER = "weatherbot - simulation only, no wallet, no order placement"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "weatherbot/0.1 (paper trading)"})
    return session


def _load(args) -> tuple[Config, Ledger]:
    cfg = Config.load(args.config)
    ledger = Ledger.load(args.ledger, starting_bankroll=cfg.bankroll)
    return cfg, ledger


def cmd_scan(args) -> int:
    cfg, ledger = _load(args)
    try:
        result = scan(cfg, ledger, session=_session())
    except MarketDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "the public Gamma and Open-Meteo endpoints must be reachable; "
            "check your network or proxy settings",
            file=sys.stderr,
        )
        return 1

    print(f"markets scanned      {result.markets_seen}")
    print(f"temperature markets  {len(result.quotes)}")
    print(f"forecasts fetched    {len(result.forecasts)}")
    print(f"signals              {len(result.signals)}")
    print(f"paper fills          {len(result.filled)}")

    for error in result.errors:
        print(f"  ! {error}", file=sys.stderr)

    if result.signals:
        print("\ntop signals")
        for signal in result.signals[:10]:
            quote = signal.quote
            print(
                f"  [{signal.side:<3}] {quote.question[:64]:<64} "
                f"price={signal.price:.3f} model={signal.model_prob:.3f} "
                f"edge={signal.edge:+.3f} stake=${signal.stake:.2f} "
                f"(fc {signal.forecast_mean:.1f}F +/-{signal.forecast_sigma:.1f})"
            )

    print()
    _print_stats(ledger)
    return 0


def cmd_settle(args) -> int:
    cfg, ledger = _load(args)
    settled, errors = settle_pending(cfg, ledger, session=_session())
    for position in settled:
        outcome = "WIN " if (position.pnl or 0) > 0 else "LOSS"
        print(
            f"  {outcome} {position.city_key} {position.target_date} "
            f"actual={position.actual_high:.1f}F "
            f"bucket=[{position.bucket_low},{position.bucket_high}] "
            f"{position.side} pnl={position.pnl:+.2f}"
        )
    for error in errors:
        print(f"  ! {error}", file=sys.stderr)
    print(f"\nsettled {len(settled)} position(s)")
    _print_stats(ledger)
    return 0


def cmd_calibrate(args) -> int:
    cfg, ledger = _load(args)
    biases = learn_bias(ledger.forecasts, min_samples=args.min_samples)
    if not biases:
        print("not enough settled history yet - run `settle` after a few days of scans")
        return 0
    for bias in biases.values():
        print("  " + bias.as_row())
    if args.apply:
        table = recalibrate(cfg, ledger, min_samples=args.min_samples)
        print(f"\napplied to config: {table}")
    else:
        print("\n(dry run - pass --apply to write these into config.json)")
    return 0


def cmd_report(args) -> int:
    _, ledger = _load(args)
    _print_stats(ledger)
    open_positions = ledger.open_positions()
    if open_positions:
        print("\nopen positions")
        for position in open_positions:
            print(
                f"  {position.city_key:<8} {position.target_date} "
                f"[{position.bucket_low},{position.bucket_high}] {position.side:<3} "
                f"@{position.price:.3f} ${position.stake:.2f}"
            )
    return 0


def cmd_loop(args) -> int:
    cfg, _ = _load(args)
    print(f"{BANNER}\nlooping every {args.interval} min - ctrl-c to stop")
    while True:
        try:
            _, ledger = _load(args)
            session = _session()
            settle_pending(cfg, ledger, session=session)
            result = scan(cfg, ledger, session=session)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"signals={len(result.signals)} fills={len(result.filled)} "
                f"equity=${ledger.equity:.2f}"
            )
        except KeyboardInterrupt:
            print("\nstopped")
            return 0
        except Exception as exc:  # keep the loop alive across transient failures
            print(f"[{time.strftime('%H:%M:%S')}] scan failed: {exc}", file=sys.stderr)
        time.sleep(args.interval * 60)


def cmd_cities(args) -> int:
    for city in CITIES.values():
        print(f"  {city.key:<8} {city.name:<16} {city.station:<22} {city.timezone}")
    return 0


def _print_stats(ledger: Ledger) -> None:
    stats = ledger.stats()
    print(
        f"equity ${stats['equity']:.2f} "
        f"(cash ${stats['cash']:.2f} + exposure ${stats['exposure']:.2f})  "
        f"return {stats['return_pct']:+.2f}%"
    )
    if stats["settled_positions"]:
        print(
            f"settled {stats['settled_positions']}  "
            f"win rate {stats['win_rate']:.1%}  "
            f"roi {stats['roi_on_settled']:+.2%}  "
            f"realised ${stats['realised_pnl']:+.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weatherbot", description=BANNER)
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--ledger", default=None, help="path to simulation.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="one forecast/pricing pass").set_defaults(func=cmd_scan)
    sub.add_parser("settle", help="score past-dated positions").set_defaults(func=cmd_settle)
    sub.add_parser("report", help="show paper PnL").set_defaults(func=cmd_report)
    sub.add_parser("cities", help="list supported cities").set_defaults(func=cmd_cities)

    calibrate = sub.add_parser("calibrate", help="learn per-city forecast bias")
    calibrate.add_argument("--apply", action="store_true", help="write results to config.json")
    calibrate.add_argument("--min-samples", type=int, default=5)
    calibrate.set_defaults(func=cmd_calibrate)

    loop = sub.add_parser("loop", help="scan and settle on an interval")
    loop.add_argument("--interval", type=int, default=30, help="minutes between passes")
    loop.set_defaults(func=cmd_loop)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
