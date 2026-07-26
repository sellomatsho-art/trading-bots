from datetime import date

import pytest

from weatherbot.config import Config
from weatherbot.forecast import EnsembleForecast
from weatherbot.ledger import Ledger, Position
from weatherbot.markets import Bucket, MarketQuote
from weatherbot.strategy import evaluate

TARGET = date(2026, 7, 26)


def make_signal(price=0.30, bucket=(90, 91), market_id="1", bankroll=1000):
    quote = MarketQuote(
        market_id=market_id,
        question="Will the highest temperature in NYC on July 26 be 90-91°F?",
        slug="s",
        city_key="nyc",
        target_date=TARGET,
        bucket=Bucket(*bucket),
        yes_price=price,
        volume=25000,
        liquidity=5000,
    )
    forecast = EnsembleForecast(
        city_key="nyc", target_date=TARGET, members=[90.0 + i * 0.05 for i in range(31)]
    )
    return evaluate(quote, forecast, Config(min_edge=0.02), bankroll)


def test_opening_a_position_moves_cash_into_exposure():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    signal = make_signal()
    position = ledger.open_position(signal)
    assert position is not None
    assert ledger.cash == pytest.approx(1000 - position.stake)
    assert ledger.exposure == pytest.approx(position.stake)
    assert ledger.equity == pytest.approx(1000), "equity is unchanged by opening at cost"


def test_a_position_larger_than_cash_is_refused():
    ledger = Ledger(starting_bankroll=1000, cash=1.0)
    assert ledger.open_position(make_signal()) is None
    assert ledger.cash == 1.0
    assert ledger.positions == []


def test_winning_yes_pays_one_dollar_per_contract():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    position = ledger.open_position(make_signal(price=0.25))
    ledger.settle("nyc", TARGET, actual_high=90.4)
    assert position.status == "settled"
    assert position.payout == pytest.approx(position.contracts)
    assert position.pnl == pytest.approx(position.contracts - position.stake)
    assert ledger.cash == pytest.approx(1000 - position.stake + position.contracts)
    assert ledger.realised_pnl > 0


def test_losing_yes_pays_nothing():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    position = ledger.open_position(make_signal(price=0.25))
    ledger.settle("nyc", TARGET, actual_high=85.0)
    assert position.payout == 0.0
    assert position.pnl == pytest.approx(-position.stake)
    assert ledger.cash == pytest.approx(1000 - position.stake)


def test_settlement_rounds_the_observation_to_the_reported_integer():
    """89.6F settles as 90, which is inside the 90-91 bucket."""
    position = Position(
        id="x", opened_at="", market_id="1", question="q", city_key="nyc",
        target_date=TARGET.isoformat(), bucket_low=90, bucket_high=91, side="YES",
        price=0.3, stake=10, contracts=33.3, model_prob=0.5, edge=0.2,
        forecast_mean=90.0, forecast_sigma=1.0,
    )
    assert position.wins_at(89.6) is True
    assert position.wins_at(89.4) is False
    assert position.wins_at(91.49) is True


def test_no_side_wins_exactly_when_yes_loses():
    position = Position(
        id="x", opened_at="", market_id="1", question="q", city_key="nyc",
        target_date=TARGET.isoformat(), bucket_low=90, bucket_high=91, side="NO",
        price=0.7, stake=10, contracts=14.3, model_prob=0.5, edge=0.2,
        forecast_mean=90.0, forecast_sigma=1.0,
    )
    assert position.wins_at(85.0) is True
    assert position.wins_at(90.0) is False


def test_settlement_only_touches_the_matching_city_and_date():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    ledger.open_position(make_signal(market_id="a"))
    other = make_signal(market_id="b")
    other.quote.city_key = "miami"
    ledger.open_position(other)

    ledger.settle("nyc", TARGET, actual_high=90.0)
    assert len(ledger.settled_positions()) == 1
    assert len(ledger.open_positions()) == 1
    assert ledger.open_positions()[0].city_key == "miami"


def test_duplicate_guard_is_per_market_and_clears_after_settlement():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    ledger.open_position(make_signal(market_id="a"))
    assert ledger.has_open_position("a") is True
    assert ledger.has_open_position("b") is False
    ledger.settle("nyc", TARGET, actual_high=90.0)
    assert ledger.has_open_position("a") is False


def test_pending_settlements_lists_past_dates_only():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    ledger.open_position(make_signal())
    ledger.record_forecast("miami", date(2026, 7, 20), 95.0, 1.2, 31)
    pending = ledger.pending_settlements(today=date(2026, 7, 27))
    assert ("nyc", TARGET) in pending
    assert ("miami", date(2026, 7, 20)) in pending
    assert ledger.pending_settlements(today=date(2026, 7, 26)) == [
        ("miami", date(2026, 7, 20))
    ]


def test_record_forecast_overwrites_the_same_city_and_date():
    ledger = Ledger()
    ledger.record_forecast("nyc", TARGET, 90.0, 1.0, 31)
    ledger.record_forecast("nyc", TARGET, 91.5, 1.2, 31)
    assert len(ledger.forecasts) == 1
    assert ledger.forecasts[0].mean == 91.5


def test_settlement_backfills_the_observation_onto_the_forecast_log():
    ledger = Ledger()
    ledger.record_forecast("nyc", TARGET, 92.0, 1.0, 31)
    ledger.settle("nyc", TARGET, actual_high=89.0)
    assert ledger.forecasts[0].actual_high == 89.0
    assert ledger.forecasts[0].error == pytest.approx(3.0), "forecast ran 3F hot"


def test_a_string_path_is_usable(tmp_path):
    """Regression: a str path reached save() uncoerced and crashed the scan."""
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=str(tmp_path / "sim.json"))
    ledger.record_forecast("nyc", TARGET, 90.0, 1.0, 31)
    ledger.save()
    assert Ledger.load(tmp_path / "sim.json").forecasts[0].mean == 90.0


def test_ledger_round_trips_through_disk(tmp_path):
    path = tmp_path / "sim.json"
    ledger = Ledger(starting_bankroll=1000, cash=1000, path=path)
    ledger.open_position(make_signal())
    ledger.record_forecast("nyc", TARGET, 90.0, 1.0, 31)
    ledger.save()

    reloaded = Ledger.load(path)
    assert reloaded.cash == pytest.approx(ledger.cash)
    assert len(reloaded.positions) == 1
    assert len(reloaded.forecasts) == 1
    assert reloaded.stats() == ledger.stats()


def test_stats_reports_win_rate_and_roi():
    ledger = Ledger(starting_bankroll=1000, cash=1000)
    ledger.open_position(make_signal(market_id="a", price=0.25))
    ledger.settle("nyc", TARGET, actual_high=90.0)
    stats = ledger.stats()
    assert stats["settled_positions"] == 1
    assert stats["win_rate"] == 1.0
    assert stats["roi_on_settled"] > 0
    assert stats["equity"] > 1000
