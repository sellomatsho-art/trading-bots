from datetime import date

import pytest

from weatherbot.cities import CITIES
from weatherbot.forecast import (
    EnsembleForecast,
    ForecastError,
    Observation,
    _local_day_bounds_utc,
    fetch_observed_high,
    parse_ensemble,
    parse_station_observations,
)

TARGET = date(2026, 7, 26)


def ensemble_payload(members=3, peak=90.0):
    """Two days of 3-hourly data shaped like the Open-Meteo ensemble response."""
    times = [f"2026-07-{day:02d}T{hour:02d}:00" for day in (26, 27) for hour in range(0, 24, 3)]
    hourly = {"time": times}
    for m in range(members):
        # Day 26 peaks at `peak + m`; day 27 peaks 10F higher so a date mix-up shows up.
        series = []
        for day in (26, 27):
            base = (peak + m) if day == 26 else (peak + m + 10)
            series.extend([base - 8, base - 6, base - 3, base, base - 1, base - 4, base - 6, base - 7])
        key = "temperature_2m" if m == 0 else f"temperature_2m_member{m:02d}"
        hourly[key] = series
    return {"hourly": hourly, "hourly_units": {"temperature_2m": "°F"}}


def test_parse_ensemble_takes_the_daily_max_per_member():
    forecast = parse_ensemble(ensemble_payload(members=3), "nyc", TARGET)
    assert forecast.member_count == 3
    assert sorted(forecast.members) == [90.0, 91.0, 92.0]


def test_parse_ensemble_reads_the_control_run_and_the_perturbed_members():
    payload = ensemble_payload(members=31)
    forecast = parse_ensemble(payload, "nyc", TARGET)
    assert forecast.member_count == 31, "control run plus 30 members"


def test_parse_ensemble_isolates_the_requested_day():
    forecast = parse_ensemble(ensemble_payload(members=1), "nyc", date(2026, 7, 27))
    assert forecast.members == [100.0]


def test_parse_ensemble_tolerates_missing_hours():
    payload = ensemble_payload(members=2)
    payload["hourly"]["temperature_2m_member01"][3] = None
    forecast = parse_ensemble(payload, "nyc", TARGET)
    assert forecast.member_count == 2


def test_parse_ensemble_rejects_a_date_outside_the_response():
    with pytest.raises(ForecastError, match="does not cover"):
        parse_ensemble(ensemble_payload(), "nyc", date(2026, 8, 15))


def test_parse_ensemble_rejects_an_empty_response():
    with pytest.raises(ForecastError, match="no hourly.time"):
        parse_ensemble({"hourly": {}}, "nyc", TARGET)


def test_mean_and_sigma():
    forecast = EnsembleForecast("nyc", TARGET, [88.0, 90.0, 92.0])
    assert forecast.mean == pytest.approx(90.0)
    assert forecast.sigma == pytest.approx(2.0)


def test_shifted_subtracts_a_hot_bias_from_every_member():
    forecast = EnsembleForecast("chicago", TARGET, [72.0, 73.0, 74.0])
    corrected = forecast.shifted(3.0)
    assert corrected.members == [69.0, 70.0, 71.0]
    assert corrected.mean == pytest.approx(70.0)
    assert forecast.members == [72.0, 73.0, 74.0], "original is untouched"


def test_shifted_by_zero_is_a_no_op():
    forecast = EnsembleForecast("nyc", TARGET, [90.0])
    assert forecast.shifted(0.0) is forecast


def test_empty_ensemble_is_an_error():
    with pytest.raises(ForecastError):
        EnsembleForecast("nyc", TARGET, [])


def _obs(stamp, celsius):
    return {"properties": {"timestamp": stamp, "temperature": {"value": celsius}}}


def test_parse_station_observations_takes_the_daily_max_in_fahrenheit():
    payload = {"features": [
        _obs("2026-07-26T14:00:00+00:00", 20.0),
        _obs("2026-07-26T20:00:00+00:00", 30.0),   # 86F, the peak
        _obs("2026-07-26T23:00:00+00:00", 25.0),
    ]}
    assert parse_station_observations(payload, TARGET, CITIES["nyc"]) == pytest.approx(86.0)


def test_parse_station_observations_uses_the_LOCAL_day():
    """A UTC day boundary would clip the west-coast afternoon peak.

    22:00 UTC on the 26th is 15:00 local in LA -- the hottest part of the
    target day, and it must be counted. 08:00 UTC on the 26th is 01:00 local
    and belongs to the same local day; 08:00 UTC on the 27th does not.
    """
    payload = {"features": [
        _obs("2026-07-26T22:00:00+00:00", 35.0),   # 15:00 PDT, in
        _obs("2026-07-27T08:00:00+00:00", 40.0),   # 01:00 PDT next day, out
    ]}
    assert parse_station_observations(payload, TARGET, CITIES["la"]) == pytest.approx(95.0)


def test_parse_station_observations_skips_null_readings():
    payload = {"features": [
        {"properties": {"timestamp": "2026-07-26T18:00:00+00:00",
                        "temperature": {"value": None}}},
        _obs("2026-07-26T19:00:00+00:00", 21.0),
    ]}
    assert parse_station_observations(payload, TARGET, CITIES["nyc"]) == pytest.approx(69.8)


def test_parse_station_observations_errors_when_the_day_has_no_reading():
    payload = {"features": [_obs("2026-07-20T18:00:00+00:00", 21.0)]}
    with pytest.raises(ForecastError, match="no usable temperature"):
        parse_station_observations(payload, TARGET, CITIES["nyc"])


def test_local_day_bounds_span_exactly_one_day():
    start, end = _local_day_bounds_utc(CITIES["la"], TARGET)
    assert (end - start).total_seconds() == 24 * 3600
    # LA is UTC-7 in July, so the local day starts at 07:00Z.
    assert start.hour == 7


def test_a_city_without_a_station_refuses_to_settle():
    """The whole point of the rewrite: never silently fall back to the grid
    the forecast came from."""
    with pytest.raises(ForecastError, match="no independent station"):
        fetch_observed_high(CITIES["london"], TARGET)


def test_observation_carries_its_provenance():
    obs = Observation(high_f=86.0, source="station", station="KLGA")
    assert obs.source == "station" and obs.station == "KLGA"
