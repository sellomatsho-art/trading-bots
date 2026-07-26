"""End-to-end scan/settle over a stubbed network.

The live endpoints are not reachable from CI, so the HTTP layer is faked at the
`requests.Session` boundary using the documented response shapes.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from weatherbot.config import Config
from weatherbot.engine import recalibrate, scan, settle_pending
from weatherbot.ledger import Ledger

TODAY = date(2026, 7, 26)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Serves ensemble, observation and Gamma responses off canned data."""

    def __init__(self, markets, peaks, observed=None):
        self.markets = markets
        self.peaks = peaks  # city_key -> forecast daily high
        self.observed = observed or {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        params = params or {}
        if "ensemble" in url:
            return FakeResponse(self._ensemble(params))
        if "daily" in params:
            return FakeResponse(self._observed(params))
        # public-search returns grouped results as a dict, not a bare list.
        # fetch_markets queries several terms and dedupes by id, so serving
        # the same payload each time is fine.
        return FakeResponse({"markets": self.markets, "events": []})

    def _city_for(self, params):
        from weatherbot.cities import CITIES

        for city in CITIES.values():
            if abs(city.latitude - params["latitude"]) < 1e-6:
                return city.key
        raise AssertionError(f"unexpected coordinates {params}")

    def _ensemble(self, params):
        peak = self.peaks[self._city_for(params)]
        times, hourly = [], {}
        days = [TODAY + timedelta(days=d) for d in range(7)]
        for day in days:
            times.extend(f"{day.isoformat()}T{hour:02d}:00" for hour in range(0, 24, 3))
        for member in range(31):
            offset = (member - 15) * 0.12  # tight, realistic ensemble spread
            series = []
            for _ in days:
                top = peak + offset
                series.extend([top - 9, top - 6, top - 3, top, top - 1, top - 4, top - 6, top - 8])
            key = "temperature_2m" if member == 0 else f"temperature_2m_member{member:02d}"
            hourly[key] = series
        hourly["time"] = times
        return {"hourly": hourly}

    def _observed(self, params):
        city = self._city_for(params)
        highs = self.observed[city]
        return {
            "daily": {
                "time": [d.isoformat() for d in sorted(highs)],
                "temperature_2m_max": [highs[d] for d in sorted(highs)],
            }
        }


def market(market_id, question, price, volume=25000):
    return {
        "id": market_id,
        "question": question,
        "slug": f"m-{market_id}",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{price}", "{round(1 - price, 4)}"]',
        "volumeNum": volume,
        "liquidityNum": 9000,
        "endDate": "2026-07-27T04:00:00Z",
    }


@pytest.fixture
def cfg():
    return Config(
        cities=["nyc", "chicago"],
        bankroll=1000.0,
        min_edge=0.05,
        min_volume=500,
        kelly_fraction=0.25,
        max_position_fraction=0.05,
        max_open_positions=3,
    )


def test_scan_prices_markets_against_the_ensemble_and_books_fills(cfg, tmp_path):
    markets = [
        # Ensemble says NYC peaks at 90.0, so the 90-91 bucket is badly underpriced.
        market("1", "Will the highest temperature in NYC on July 26 be 90-91°F?", 0.20),
        # ...and the 95F+ bucket is badly overpriced.
        market("2", "Will the highest temperature in NYC on July 26 be 95°F or above?", 0.40),
    ]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")

    result = scan(cfg, ledger, session=session, today=TODAY)

    assert result.markets_seen == 2
    assert len(result.quotes) == 2
    assert len(result.forecasts) == 1, "one city/date pair needed a forecast"
    sides = {s.quote.market_id: s.side for s in result.signals}
    assert sides == {"1": "YES", "2": "NO"}
    assert len(result.filled) == 2
    assert ledger.equity == pytest.approx(1000), "opening at cost does not change equity"
    assert ledger.cash < 1000


def test_scan_ignores_cities_outside_the_config_and_dates_beyond_the_horizon(cfg, tmp_path):
    far = (TODAY + timedelta(days=30)).strftime("%B %-d")
    markets = [
        market("1", "Will the highest temperature in Miami on July 26 be 90-91°F?", 0.20),
        market("2", f"Will the highest temperature in NYC on {far} be 90-91°F?", 0.20),
    ]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")

    result = scan(cfg, ledger, session=session, today=TODAY)
    assert result.quotes == []
    assert result.filled == []


def test_scan_respects_the_open_position_cap(cfg, tmp_path):
    cfg.max_open_positions = 1
    markets = [
        market("1", "Will the highest temperature in NYC on July 26 be 90-91°F?", 0.20),
        market("2", "Will the highest temperature in NYC on July 26 be 95°F or above?", 0.40),
    ]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")

    result = scan(cfg, ledger, session=session, today=TODAY)
    assert len(result.signals) == 2
    assert len(result.filled) == 1, "cap applies even when more edges are available"


def test_rescanning_does_not_double_up_on_the_same_market(cfg, tmp_path):
    markets = [market("1", "Will the highest temperature in NYC on July 26 be 90-91°F?", 0.20)]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")

    scan(cfg, ledger, session=session, today=TODAY)
    second = scan(cfg, ledger, session=session, today=TODAY)
    assert second.filled == []
    assert len(ledger.open_positions()) == 1


def test_a_forecast_failure_skips_the_city_without_killing_the_scan(cfg, tmp_path):
    class BrokenEnsemble(FakeSession):
        def _ensemble(self, params):
            return {"hourly": {}}

    markets = [market("1", "Will the highest temperature in NYC on July 26 be 90-91°F?", 0.20)]
    session = BrokenEnsemble(markets, peaks={"nyc": 90.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")

    result = scan(cfg, ledger, session=session, today=TODAY)
    assert result.errors and "hourly.time" in result.errors[0]
    assert result.filled == []


def test_settle_scores_positions_against_the_observed_high(cfg, tmp_path):
    markets = [market("1", "Will the highest temperature in NYC on July 26 be 90-91°F?", 0.20)]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")
    scan(cfg, ledger, session=session, today=TODAY)

    session.observed = {"nyc": {TODAY: 90.3}}
    settled, errors = settle_pending(cfg, ledger, session=session, today=TODAY + timedelta(days=1))

    assert errors == []
    assert len(settled) == 1
    assert settled[0].pnl > 0, "bought the 90-91 bucket at 0.20 and it hit"
    assert ledger.equity > 1000
    assert ledger.open_positions() == []


def test_a_missed_forecast_costs_the_stake(cfg, tmp_path):
    markets = [market("1", "Will the highest temperature in NYC on July 26 be 90-91°F?", 0.20)]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")
    scan(cfg, ledger, session=session, today=TODAY)
    staked = ledger.exposure

    session.observed = {"nyc": {TODAY: 84.0}}
    settled, _ = settle_pending(cfg, ledger, session=session, today=TODAY + timedelta(days=1))

    assert settled[0].pnl == pytest.approx(-staked)
    assert ledger.equity == pytest.approx(1000 - staked)


def test_the_full_loop_learns_a_bias_and_writes_it_back(cfg, tmp_path):
    """Forecast 90F, observe 87F, repeatedly -> the model should learn it runs hot."""
    cfg_path = tmp_path / "config.json"
    cfg.save(cfg_path)
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "sim.json")
    session = FakeSession([], peaks={"nyc": 90.0, "chicago": 80.0})

    for offset in range(12):
        day = TODAY - timedelta(days=offset + 1)
        ledger.record_forecast("nyc", day, 90.0, 1.0, 31)
        ledger.settle("nyc", day, actual_high=87.0)

    from weatherbot.config import Config as C

    reloaded = C.load(cfg_path)
    reloaded.bias_correction = {}
    applied = recalibrate(reloaded, ledger, min_samples=5)

    assert applied["nyc"] == pytest.approx(3.0 * 12 / 22, abs=0.01)
    assert C.load(cfg_path).bias_correction["nyc"] == pytest.approx(applied["nyc"])


def test_bias_correction_shifts_the_probabilities_a_scan_produces(cfg, tmp_path):
    """With a 5F hot bias applied, a 90F raw forecast prices as 85F.

    The same market flips sides: uncorrected, the 84-85 bucket looks nearly
    impossible and the bot sells it; corrected, it is the modal outcome.
    """
    markets = [market("1", "Will the highest temperature in NYC on July 26 be 84-85°F?", 0.20)]
    session = FakeSession(markets, peaks={"nyc": 90.0, "chicago": 80.0})

    uncorrected = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "a.json")
    raw = scan(cfg, uncorrected, session=session, today=TODAY)
    assert raw.signals[0].side == "NO"
    assert raw.signals[0].forecast_mean == pytest.approx(90.0, abs=0.1)

    cfg.bias_correction = {"nyc": 5.0}
    corrected = Ledger(starting_bankroll=1000, cash=1000, path=tmp_path / "b.json")
    result = scan(cfg, corrected, session=session, today=TODAY)

    assert len(result.filled) == 1
    assert result.signals[0].side == "YES"
    assert result.signals[0].forecast_mean == pytest.approx(85.0, abs=0.1)
