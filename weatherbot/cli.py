"""Command line entry point.

    python -m weatherbot scan        one pass: forecast, price, book paper fills
    python -m weatherbot settle      score past-dated positions against observations
    python -m weatherbot calibrate   learn per-city forecast bias from history
    python -m weatherbot report      current paper PnL
    python -m weatherbot scalp       post-event scalp candidates for today
    python -m weatherbot reset       archive the ledger and start clean
    python -m weatherbot loop        scan + settle on an interval
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from .calibration import learn_bias
from .cities import CITIES
from .config import Config
from .engine import recalibrate, scan, settle_pending
from .forecast import ForecastError
from .ledger import DEFAULT_LEDGER_PATH, Ledger
from .markets import MarketDataError, fetch_markets, temperature_quotes
from .scalp import bucket_probability, fetch_scalp_inputs

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

    if result.uncalibrated:
        print("\nheld back - no measured bias yet")
        for city_key, seen, needed in result.uncalibrated:
            print(f"  {city_key:<8} {seen}/{needed} station observations"
                  f"  (forecast still logged; run `settle` after the date passes)")

    if result.signals:
        print("\ntop signals")
        for signal in result.signals[:10]:
            quote = signal.quote
            bucket = quote.bucket
            lo = "-inf" if bucket.low == float("-inf") else f"{bucket.low:.0f}"
            hi = "inf" if bucket.high == float("inf") else f"{bucket.high:.0f}"
            print(
                f"  [{signal.side:<3}] {quote.city_key:<8} {quote.target_date} "
                f"[{lo},{hi}]F  price={signal.price:.3f} model={signal.model_prob:.3f} "
                f"edge={signal.edge:+.3f} stake=${signal.stake:.2f} "
                f"(fc {signal.forecast_mean:.1f}F +/-{signal.forecast_sigma:.1f})"
            )
            # Full text, untruncated: the tail of the question carries the
            # bucket and the date, and cutting it hid a real diagnosis.
            print(f"          {quote.question}")

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


def cmd_scalp(args) -> int:
    """Report where today's observed max already contradicts the market.

    Report-only on purpose. The cheapest possible test of the hypothesis is
    to look once at whether the gaps exist at all; wiring it to the ledger
    before knowing that would be building on a guess, which is the mistake
    the last two strategies made.
    """
    cfg, _ = _load(args)
    session = _session()
    today = date.today()

    try:
        quotes = [q for q in temperature_quotes(fetch_markets(session), today)
                  if q.target_date == today and q.city_key in set(cfg.cities)]
    except MarketDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not quotes:
        print("no temperature markets settling today for the configured cities")
        return 0

    by_city: dict[str, list] = {}
    for q in quotes:
        by_city.setdefault(q.city_key, []).append(q)

    candidates = 0
    for city_key in sorted(by_city):
        city = CITIES[city_key]
        try:
            state, dist = fetch_scalp_inputs(
                city, today, cfg.scalp_cutoff_hour, cfg.scalp_history_days,
                session=session)
        except ForecastError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            continue

        print(f"\n{city.name} ({city.station_id})")
        if state is None:
            print(f"  nothing reported before {cfg.scalp_cutoff_hour}:00 local yet")
            continue
        print(f"  max so far {state.max_so_far:.1f}F (rounds to {state.rounded_max}) "
              f"from {state.readings} readings, last {state.last_reading_at:%H:%M} local")

        if dist.samples < cfg.scalp_min_samples:
            print(f"  late-rise risk unmeasured: {dist.samples}/{cfg.scalp_min_samples} "
                  f"past days - not pricing anything")
            continue
        spread = ", ".join(f"+{k}F {dist.probability(k):.1%}"
                           for k in range(0, dist.max_rise + 2))
        print(f"  late rise over {dist.samples} past days: {spread}")

        for q in sorted(by_city[city_key], key=lambda x: x.bucket.low):
            model = bucket_probability(state, q.bucket, dist)
            for side, price, prob in (("YES", q.yes_price, model),
                                      ("NO", 1 - q.yes_price, 1 - model)):
                edge = prob - price
                if edge < cfg.scalp_min_edge or not 0.02 <= price <= 0.98:
                    continue
                lo = "-inf" if q.bucket.low == float("-inf") else f"{q.bucket.low:.0f}"
                hi = "inf" if q.bucket.high == float("inf") else f"{q.bucket.high:.0f}"
                print(f"    [{side:<3}] [{lo},{hi}]F  market={price:.3f} "
                      f"model={prob:.3f}  edge={edge:+.3f}")
                candidates += 1

    print(f"\n{candidates} candidate(s) above {cfg.scalp_min_edge:.0%} edge")
    print("Report only - nothing staked. Quoted price is not executable price;\n"
          "depth is not modelled, and settlement source is assumed to be the\n"
          "station max rather than the 6-hourly climate report.")
    return 0


def cmd_reset(args) -> int:
    """Archive the paper ledger and start a fresh one.

    Archives rather than deletes: a settled record is data, and the old one
    is the evidence for whatever made you reset. The bias history is the
    reason this exists -- positions settled against the model grid carry the
    grid's own error, so an equity curve built on them is not a measurement
    of anything and should not be carried into a clean run.

    Note this is about the PnL record only. The calibration gate is already
    safe on its own: observed_counts() requires source="station", so
    grid-settled or legacy rows never open it.
    """
    cfg, ledger = _load(args)
    path = Path(ledger.path or DEFAULT_LEDGER_PATH)
    if not path.exists():
        print(f"nothing to reset: {path} does not exist")
        return 0

    stats = ledger.stats()
    observed = ledger.observed_counts()
    print(f"{path}")
    print(f"  {stats['settled_positions']} settled, {stats['open_positions']} open, "
          f"equity ${stats['equity']:.2f}")
    print(f"  {len(ledger.forecasts)} forecast records, "
          f"{sum(observed.values())} scored against a station")

    if not args.yes:
        if not sys.stdin.isatty():
            print("refusing to reset non-interactively; pass --yes", file=sys.stderr)
            return 1
        if input("archive and start fresh? [y/N] ").strip().lower() not in ("y", "yes"):
            print("cancelled")
            return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    path.rename(archive)
    Ledger(starting_bankroll=cfg.bankroll, cash=cfg.bankroll, path=path).save()
    print(f"\narchived to {archive.name}")
    print(f"fresh ledger at ${cfg.bankroll:.2f}")
    return 0


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

    sub.add_parser("scalp", help="report post-event scalp candidates for today"
                   ).set_defaults(func=cmd_scalp)

    reset = sub.add_parser("reset", help="archive the ledger and start fresh")
    reset.add_argument("--yes", action="store_true", help="skip the confirmation")
    reset.set_defaults(func=cmd_reset)

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
