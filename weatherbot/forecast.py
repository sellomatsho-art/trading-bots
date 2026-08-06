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
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from .cities import City

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
#: Official station observations. Free, no key, and -- the point -- produced
#: independently of the model the forecast comes from.
NWS_STATIONS_URL = "https://api.weather.gov/stations"
NWS_HEADERS = {"User-Agent": "weatherbot (paper trading research)",
               "Accept": "application/geo+json"}
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


@dataclass(frozen=True)
class Observation:
    """A settled daily high, with where it came from.

    Provenance is not decoration. Settling against the same Open-Meteo grid
    the forecast is drawn from means a systematic grid error cancels on both
    sides and the calibration loop cannot see it -- measured live on 3 Aug
    2026, when the ensemble sat ~5F above the market at both LAX and LGA and
    the bot was still grading itself as roughly right. Only `station`
    observations are independent enough to learn a bias from.
    """

    high_f: float
    source: str  # "station" or "grid"
    station: str | None = None


def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _local_day_bounds_utc(city: City, target_date: date) -> tuple[datetime, datetime]:
    """The city's local calendar day, expressed in UTC.

    Markets settle on the local calendar day, so the window has to be built
    in the city's zone and converted -- a UTC day boundary would clip the
    afternoon peak on the US west coast.
    """
    tz = ZoneInfo(city.timezone)
    start = datetime.combine(target_date, time.min, tzinfo=tz)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def parse_station_series(payload: dict, city: City) -> list[tuple[datetime, float]]:
    """Every usable reading in the payload as (local time, degrees F).

    Returned in the city's own zone and sorted, because the scalp needs to
    ask "what was the max as of 6pm local" -- a question about the shape of
    the day, not just its peak.
    """
    tz = ZoneInfo(city.timezone)
    out: list[tuple[datetime, float]] = []
    for feature in payload.get("features") or []:
        props = (feature or {}).get("properties") or {}
        temp = (props.get("temperature") or {}).get("value")
        stamp = props.get("timestamp")
        if temp is None or not stamp:
            continue
        try:
            when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append((when.astimezone(tz), _c_to_f(float(temp))))
    out.sort(key=lambda r: r[0])
    return out


def parse_station_observations(payload: dict, target_date: date, city: City) -> float:
    """Highest temperature reported by the station during the local day."""
    highs = [t for when, t in parse_station_series(payload, city)
             if when.date() == target_date]
    if not highs:
        raise ForecastError(
            f"station {city.station_id} reported no usable temperature for "
            f"{target_date.isoformat()}"
        )
    return max(highs)


def fetch_station_series(
    city: City,
    start_date: date,
    end_date: date,
    session: requests.Session | None = None,
    chunk_days: int = 7,
) -> list[tuple[datetime, float]]:
    """Readings across a local date range, inclusive.

    Chunked because NWS caps `limit` at 500 and a station reports roughly
    30 readings a day once specials are counted, so a month in one call
    would silently truncate -- and a truncated history would quietly bias
    the late-rise measurement toward whichever end of the window survived.
    """
    if not city.station_id:
        raise ForecastError(f"no independent station for {city.key}")
    http = session or requests
    out: list[tuple[datetime, float]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        start, _ = _local_day_bounds_utc(city, cursor)
        _, end = _local_day_bounds_utc(city, chunk_end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 500,
        }
        url = f"{NWS_STATIONS_URL}/{city.station_id}/observations"
        try:
            resp = http.get(url, params=params, timeout=TIMEOUT, headers=NWS_HEADERS)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ForecastError(
                f"station history fetch failed for {city.station_id}: {exc}"
            ) from exc
        out.extend(parse_station_series(resp.json(), city))
        cursor = chunk_end + timedelta(days=1)
    out.sort(key=lambda r: r[0])
    return out


def fetch_observed_high(
    city: City,
    target_date: date,
    session: requests.Session | None = None,
) -> Observation:
    """Read the realised daily high from the city's official station.

    Uses the NWS observation feed, which is free, key-free and independent of
    the model the forecast comes from. That independence is the whole point:
    a bias the forecast and the settlement source share is invisible to
    calibration.

    Highs are taken as the max of the hourly/METAR reports over the local day,
    which tracks but does not always exactly equal the official climate-report
    max (that uses 6-hourly max/min groups). Close enough to score a paper
    book; still not the resolution source Polymarket itself uses.

    Cities with no station feed raise rather than silently falling back to the
    grid -- see Observation.
    """
    if not city.station_id:
        raise ForecastError(
            f"no independent station for {city.key}; refusing to settle against "
            "the same grid the forecast came from"
        )
    http = session or requests
    start, end = _local_day_bounds_utc(city, target_date)
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 500,
    }
    url = f"{NWS_STATIONS_URL}/{city.station_id}/observations"
    try:
        resp = http.get(url, params=params, timeout=TIMEOUT, headers=NWS_HEADERS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ForecastError(f"station fetch failed for {city.station_id}: {exc}") from exc
    high = parse_station_observations(resp.json(), target_date, city)
    return Observation(high_f=high, source="station", station=city.station_id)
