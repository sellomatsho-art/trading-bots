from datetime import date

import pytest

from weatherbot.config import Config
from weatherbot.forecast import EnsembleForecast
from weatherbot.markets import Bucket, MarketQuote
from weatherbot.strategy import (
    bucket_probability,
    empirical_probability,
    evaluate,
    find_signals,
    gaussian_probability,
    kelly_fraction,
)

TARGET = date(2026, 7, 26)


def forecast(members):
    return EnsembleForecast(city_key="nyc", target_date=TARGET, members=list(members))


def quote(price=0.30, bucket=(90, 91), volume=25000, market_id="1"):
    return MarketQuote(
        market_id=market_id,
        question="Will the highest temperature in NYC on July 26 be 90-91°F?",
        slug="s",
        city_key="nyc",
        target_date=TARGET,
        bucket=Bucket(*bucket),
        yes_price=price,
        volume=volume,
        liquidity=5000,
    )


def test_empirical_probability_counts_members_in_the_widened_bucket():
    # 89.6 and 91.4 round into the 90-91 bucket; 89.2 and 91.9 do not.
    members = [89.2, 89.6, 90.4, 91.4, 91.9]
    assert empirical_probability(members, Bucket(90, 91)) == pytest.approx(3 / 5)


def test_empirical_probability_handles_open_ended_buckets():
    members = [88.0, 92.0, 95.0, 99.0]
    assert empirical_probability(members, Bucket(94, float("inf"))) == pytest.approx(0.5)
    assert empirical_probability(members, Bucket(float("-inf"), 89)) == pytest.approx(0.25)


def test_gaussian_probability_is_symmetric_around_the_mean():
    members = [90.0] * 30
    p = gaussian_probability(members, Bucket(90, 90), min_sigma=1.5)
    below = gaussian_probability(members, Bucket(float("-inf"), 89), min_sigma=1.5)
    above = gaussian_probability(members, Bucket(91, float("inf")), min_sigma=1.5)
    assert below == pytest.approx(above, abs=1e-9)
    assert p + below + above == pytest.approx(1.0, abs=1e-9)


def test_min_sigma_floor_stops_a_tight_ensemble_claiming_certainty():
    members = [90.0] * 31
    tight = gaussian_probability(members, Bucket(90, 90), min_sigma=1.5)
    assert 0.2 < tight < 0.45  # nowhere near the 1.0 the raw histogram would give
    assert empirical_probability(members, Bucket(90, 90)) == 1.0


def test_bucket_probability_clamps_away_from_certainty():
    cfg = Config(probability_model="empirical")
    p = bucket_probability(forecast([70.0] * 31), Bucket(90, 91), cfg)
    assert p == pytest.approx(0.01)


def test_blend_sits_between_its_two_components():
    cfg = Config(probability_model="blend", empirical_weight=0.5, min_sigma_f=1.5)
    members = [88.5, 89.0, 90.0, 90.5, 91.0, 92.0, 93.0]
    bucket = Bucket(90, 91)
    fc = forecast(members)
    emp = empirical_probability(members, bucket)
    gauss = gaussian_probability(members, bucket, 1.5)
    assert bucket_probability(fc, bucket, cfg) == pytest.approx(0.5 * emp + 0.5 * gauss)
    assert min(emp, gauss) <= bucket_probability(fc, bucket, cfg) <= max(emp, gauss)


@pytest.mark.parametrize(
    "prob,price,expected",
    [
        (0.60, 0.40, 1 / 3),
        (0.40, 0.40, 0.0),
        (0.30, 0.40, 0.0),  # negative edge is never staked
        (0.99, 0.01, pytest.approx(0.98 / 0.99)),
    ],
)
def test_kelly_fraction(prob, price, expected):
    assert kelly_fraction(prob, price) == pytest.approx(expected)


def test_kelly_fraction_rejects_degenerate_prices():
    assert kelly_fraction(0.6, 0.0) == 0.0
    assert kelly_fraction(0.6, 1.0) == 0.0


def test_evaluate_buys_yes_when_the_model_is_above_the_market():
    cfg = Config(min_edge=0.05, kelly_fraction=0.25, max_position_fraction=0.05)
    fc = forecast([90.0 + i * 0.05 for i in range(31)])  # tight around the bucket
    signal = evaluate(quote(price=0.30), fc, cfg, bankroll=1000)
    assert signal is not None
    assert signal.side == "YES"
    assert signal.edge > 0.05
    assert signal.stake <= 1000 * cfg.max_position_fraction
    assert signal.contracts == pytest.approx(signal.stake / signal.price, rel=1e-3)


def test_evaluate_buys_no_when_the_market_is_above_the_model():
    cfg = Config(min_edge=0.05)
    fc = forecast([70.0 + i * 0.1 for i in range(31)])  # nowhere near 90-91
    signal = evaluate(quote(price=0.80), fc, cfg, bankroll=1000)
    assert signal is not None
    assert signal.side == "NO"
    assert signal.price == pytest.approx(0.20)
    assert signal.model_prob > 0.9


def test_evaluate_respects_edge_price_and_volume_filters():
    cfg = Config(min_edge=0.05, min_price=0.05, max_price=0.95, min_volume=500)
    fc = forecast([90.0 + i * 0.05 for i in range(31)])
    assert evaluate(quote(price=0.30, volume=10), fc, cfg, 1000) is None, "thin market"
    assert evaluate(quote(price=0.99), fc, cfg, 1000) is None, "outside price band"
    cfg_strict = Config(min_edge=0.99)
    assert evaluate(quote(price=0.30), fc, cfg_strict, 1000) is None, "edge below threshold"


def test_expected_value_is_positive_on_a_taken_signal():
    cfg = Config(min_edge=0.05)
    fc = forecast([90.0 + i * 0.05 for i in range(31)])
    signal = evaluate(quote(price=0.30), fc, cfg, bankroll=1000)
    assert signal.expected_value > 0


def test_find_signals_sorts_by_edge_and_skips_unforecast_markets():
    cfg = Config(min_edge=0.02)
    fc = forecast([90.0 + i * 0.05 for i in range(31)])
    quotes = [
        quote(price=0.30, market_id="a"),
        quote(price=0.45, market_id="b"),
        MarketQuote(
            market_id="c",
            question="Will the highest temperature in Miami on July 26 be 90-91°F?",
            slug="s",
            city_key="miami",
            target_date=TARGET,
            bucket=Bucket(90, 91),
            yes_price=0.3,
            volume=25000,
            liquidity=5000,
        ),
    ]
    signals = find_signals(quotes, {("nyc", TARGET.isoformat()): fc}, cfg, bankroll=1000)
    assert [s.quote.market_id for s in signals] == ["a", "b"]
    assert signals[0].edge >= signals[1].edge
