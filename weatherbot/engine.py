"""Scan / settle orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import requests

from .calibration import bias_table, learn_bias
from .cities import CITIES
from .config import Config
from .forecast import EnsembleForecast, ForecastError, fetch_ensemble, fetch_observed_high
from .ledger import Ledger, Position
from .markets import MarketQuote, fetch_markets, temperature_quotes
from .strategy import Signal, find_signals


@dataclass
class ScanResult:
    markets_seen: int = 0
    quotes: list[MarketQuote] = field(default_factory=list)
    forecasts: dict[tuple[str, str], EnsembleForecast] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    filled: list[Position] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def scan(
    cfg: Config,
    ledger: Ledger,
    session: requests.Session | None = None,
    today: date | None = None,
) -> ScanResult:
    """One full pass: read markets, forecast, price, book paper fills."""
    today = today or date.today()
    result = ScanResult()

    raw = fetch_markets(session)
    result.markets_seen = len(raw)

    horizon = today + timedelta(days=cfg.forecast_horizon_days)
    wanted = set(cfg.cities)
    result.quotes = [
        q
        for q in temperature_quotes(raw, today)
        if q.city_key in wanted and today <= q.target_date <= horizon
    ]

    needed = sorted({(q.city_key, q.target_date) for q in result.quotes})
    for city_key, target_date in needed:
        city = CITIES.get(city_key)
        if city is None:
            continue
        try:
            forecast = fetch_ensemble(city, target_date, session=session)
        except ForecastError as exc:
            result.errors.append(str(exc))
            continue
        ledger.record_forecast(
            city_key, target_date, forecast.mean, forecast.sigma, forecast.member_count
        )
        bias = cfg.bias_correction.get(city_key, 0.0)
        result.forecasts[(city_key, target_date.isoformat())] = forecast.shifted(bias)

    result.signals = find_signals(result.quotes, result.forecasts, cfg, ledger.equity)

    room = cfg.max_open_positions - len(ledger.open_positions())
    for signal in result.signals:
        if room <= 0:
            break
        if ledger.has_open_position(signal.quote.market_id):
            continue
        position = ledger.open_position(signal)
        if position is not None:
            result.filled.append(position)
            room -= 1

    ledger.save()
    return result


def settle_pending(
    cfg: Config,
    ledger: Ledger,
    session: requests.Session | None = None,
    today: date | None = None,
) -> tuple[list[Position], list[str]]:
    """Score every past-dated position against the observed high."""
    today = today or date.today()
    settled: list[Position] = []
    errors: list[str] = []

    for city_key, target_date in ledger.pending_settlements(today):
        city = CITIES.get(city_key)
        if city is None:
            errors.append(f"unknown city {city_key!r} in ledger")
            continue
        try:
            actual = fetch_observed_high(city, target_date, session=session)
        except ForecastError as exc:
            errors.append(str(exc))
            continue
        settled.extend(ledger.settle(city_key, target_date, actual))

    ledger.save()
    return settled, errors


def recalibrate(cfg: Config, ledger: Ledger, min_samples: int = 5) -> dict[str, float]:
    """Refresh per-city bias from settled forecasts and persist it to config."""
    biases = learn_bias(ledger.forecasts, min_samples=min_samples)
    table = bias_table(biases)
    if table:
        cfg.bias_correction.update(table)
        cfg.save()
    return table
