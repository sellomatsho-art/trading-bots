"""GFS ensemble forecasts via Open-Meteo (free, no API key, no registration).

The ensemble endpoint returns every GFS member separately -- for `gfs025` that
is the control run plus 30 perturbed members, i.e. the 31 scenarios. The spread
across those members is the whole point: it is a direct, per-city estimate of
how uncertain tomorrow's high actually is, which is exactly what a bucketed
temperature market is pricing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

from .cities import City

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "gfs025"
TIMEOUT = 30


class ForecastError(RuntimeError):
    """Raised when upstream weather data is missing or unusable."""


@dataclass
class EnsembleForecast:
    city_key: str
    target_date: date
    members: list[float]
    model: str = DEFAULT_MODEL
    fetched_at: str = ""

    def __post_init__(self) -> None:
        if not self.members:
            raise ForecastError(f"empty ensemble for {self.city_key} on {self.target_date}")
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def mean(self) -> float:
        return statistics.fmean(self.members)

    @property
    def sigma(self) -> float:
        return statistics.stdev(self.members) if len(self.members) > 1 else 0.0

    @property
    def member_count(self) -> int:
        return len(self.members)

    def shifted(self, bias_f: float) -> "EnsembleForecast":
        """Return a copy with a learned bias removed from every member.

        `bias_f` is how much this model has historically run *hot* for the
        city, so it is subtracted.
        """
        if not bias_f:
            return self
        return EnsembleForecast(
            city_key=self.city_key,
            target_date=self.target_date,
            members=[m - bias_f for m in self.members],
            model=self.model,
            fetched_at=self.fetched_at,
        )


def parse_ensemble(payload: dict, city_key: str, target_date: date, model: str = DEFAULT_MODEL) -> EnsembleForecast:
    """Reduce an hourly ensemble response to one daily high per member.

    Open-Meteo names the control run `temperature_2m` and the perturbed members
    `temperature_2m_member01`..`memberNN`. Both forms are handled.
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    if not times:
        raise ForecastError("ensemble response has no hourly.time series")

    stamp = target_date.isoformat()
    idx = [i for i, t in enumerate(times) if t.startswith(stamp)]
    if not idx:
        raise ForecastError(f"ensemble response does not cover {stamp}")

    members: list[float] = []
    for key, series in hourly.items():
        if key == "time" or not key.startswith("temperature_2m"):
            continue
        day = [series[i] for i in idx if i < len(series) and series[i] is not None]
        if day:
            members.append(max(day))

    if not members:
        raise ForecastError(f"no usable members for {city_key} on {stamp}")
    return EnsembleForecast(city_key=city_key, target_date=target_date, members=members, model=model)


def fetch_ensemble(
    city: City,
    target_date: date,
    model: str = DEFAULT_MODEL,
    session: requests.Session | None = None,
) -> EnsembleForecast:
    """Fetch the GFS ensemble and reduce it to per-member daily highs (F)."""
    http = session or requests
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "models": model,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": city.timezone,
        "forecast_days": 7,
    }
    try:
        resp = http.get(ENSEMBLE_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ForecastError(f"ensemble fetch failed for {city.key}: {exc}") from exc
    return parse_ensemble(resp.json(), city.key, target_date, model)


def parse_observed_high(payload: dict, target_date: date) -> float:
    """Pull the observed daily high for `target_date` out of a daily response."""
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    stamp = target_date.isoformat()
    for t, high in zip(times, highs):
        if t == stamp and high is not None:
            return float(high)
    raise ForecastError(f"no observed high available for {stamp}")


def fetch_observed_high(
    city: City,
    target_date: date,
    session: requests.Session | None = None,
) -> float:
    """Fetch the realised daily high used to settle paper positions.

    This is the reanalysis/observation blend for the city's coordinates, not
    the official station report Polymarket resolves against. It is close enough
    to score a simulation and is what keeps the whole pipeline key-free; treat
    a settled paper PnL as indicative, not exact.
    """
    http = session or requests
    days_back = (date.today() - target_date).days
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": city.timezone,
        "past_days": max(1, min(days_back + 1, 92)),
        "forecast_days": 1,
    }
    try:
        resp = http.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ForecastError(f"observation fetch failed for {city.key}: {exc}") from exc
    return parse_observed_high(resp.json(), target_date)
