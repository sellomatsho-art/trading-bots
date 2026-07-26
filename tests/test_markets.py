from datetime import date

import pytest

import requests

from weatherbot.markets import (
    Bucket,
    MarketDataError,
    build_quote,
    fetch_markets,
    parse_bucket,
    parse_target_date,
    temperature_quotes,
)

TODAY = date(2026, 7, 26)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Will the highest temperature in NYC on July 26 be 90-91°F?", (90, 91)),
        ("Highest temperature in Chicago on July 26: 88–89F", (88, 89)),
        ("Will the high temperature in Austin be 100 to 102 degrees?", (100, 102)),
        ("Will the highest temperature in Miami on July 26 be 94°F or above?", (94, float("inf"))),
        ("Will the highest temperature in Miami be at least 94 degrees?", (94, float("inf"))),
        ("Highest temperature in Denver: 95F+", (95, float("inf"))),
        ("Will the highest temperature in NYC be 89°F or below?", (float("-inf"), 89)),
        ("Will the highest temperature in NYC be at most 89 degrees?", (float("-inf"), 89)),
    ],
)
def test_parse_bucket_inclusive_forms(question, expected):
    assert tuple(parse_bucket(question)) == expected


def test_strict_bounds_exclude_the_threshold():
    """"below 90" settles at 89 or cooler; "above 90" settles at 91 or hotter."""
    assert tuple(parse_bucket("Will the temperature in NYC be below 90°F?")) == (float("-inf"), 89)
    assert tuple(parse_bucket("Will the temperature in NYC be above 90°F?")) == (91, float("inf"))
    assert tuple(parse_bucket("Will the temperature be higher than 100 degrees?")) == (101, float("inf"))


def test_parse_bucket_returns_none_for_non_bucket_text():
    assert parse_bucket("Will it rain in Seattle tomorrow?") is None


def test_continuous_bounds_widen_integer_buckets():
    assert Bucket(90, 91).continuous_bounds() == (89.5, 91.5)
    assert Bucket(94, float("inf")).continuous_bounds() == (93.5, float("inf"))
    assert Bucket(float("-inf"), 89).continuous_bounds() == (float("-inf"), 89.5)


def test_parse_target_date_from_question_and_slug():
    assert parse_target_date("Highest temperature in NYC on July 26?", TODAY) == date(2026, 7, 26)
    assert parse_target_date("highest-temperature-nyc-august-3", TODAY) == date(2026, 8, 3)
    assert parse_target_date("high temp on January 4, 2027", TODAY) == date(2027, 1, 4)


def test_parse_target_date_rolls_to_next_year_across_the_boundary():
    december = date(2026, 12, 28)
    assert parse_target_date("Highest temperature in NYC on January 2?", december) == date(2027, 1, 2)


def _raw(**overrides):
    raw = {
        "id": "512345",
        "question": "Will the highest temperature in NYC on July 26 be 90-91°F?",
        "slug": "highest-temperature-in-nyc-on-july-26-90-91",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.31", "0.69"]',
        "volumeNum": 25000,
        "liquidityNum": 8000,
        "endDate": "2026-07-27T04:00:00Z",
    }
    raw.update(overrides)
    return raw


def test_build_quote_extracts_everything_needed_to_price():
    quote = build_quote(_raw(), TODAY)
    assert quote is not None
    assert quote.city_key == "nyc"
    assert quote.target_date == date(2026, 7, 26)
    assert tuple(quote.bucket) == (90, 91)
    assert quote.yes_price == pytest.approx(0.31)
    assert quote.volume == 25000


def test_build_quote_reads_the_yes_leg_even_when_listed_second():
    quote = build_quote(_raw(outcomes='["No", "Yes"]', outcomePrices='["0.69", "0.31"]'), TODAY)
    assert quote.yes_price == pytest.approx(0.31)


def test_build_quote_skips_non_temperature_and_unknown_cities():
    assert build_quote(_raw(question="Will BTC close above $100k?"), TODAY) is None
    assert build_quote(_raw(question="Will the temperature in Reykjavik be 50-51°F?", slug="x"), TODAY) is None


def test_temperature_quotes_filters_a_mixed_feed():
    feed = [_raw(), {"question": "Will BTC hit 200k?", "outcomes": '["Yes","No"]'}, {"question": ""}]
    assert len(temperature_quotes(feed, TODAY)) == 1


def test_a_network_failure_surfaces_as_a_typed_error_not_a_traceback():
    class DeadSession:
        def get(self, *a, **kw):
            raise requests.ConnectionError("tunnel refused")

    with pytest.raises(MarketDataError, match="could not reach"):
        fetch_markets(DeadSession())


def test_fetch_markets_queries_search_dedupes_and_drops_closed():
    """Weather markets are too low-volume for the volume-ordered listing,
    which also 422s past an offset ceiling. fetch_markets uses search instead:
    several terms, results deduped by id, settled markets discarded."""

    class SearchSession:
        def __init__(self):
            self.urls = []
            self.terms = []

        def get(self, url, params=None, timeout=None):
            self.urls.append(url)
            self.terms.append(params["q"])

            class R:
                @staticmethod
                def raise_for_status():
                    return None

                @staticmethod
                def json():
                    return {
                        "markets": [_raw()],
                        "events": [
                            {"markets": [_raw(id="999", closed=True)]},
                        ],
                    }

            return R()

    session = SearchSession()
    out = fetch_markets(session)

    assert len(out) == 1, "one live market, deduped across terms; closed one dropped"
    assert out[0]["id"] == "512345"
    assert all("public-search" in url for url in session.urls)
    assert len(session.terms) > 1, "more than one search term is tried"


def test_fetch_markets_keeps_markets_nested_under_events():
    class EventOnlySession:
        def get(self, url, params=None, timeout=None):
            class R:
                @staticmethod
                def raise_for_status():
                    return None

                @staticmethod
                def json():
                    return {"events": [{"markets": [_raw()]}]}

            return R()

    assert len(fetch_markets(EventOnlySession())) == 1


def test_build_quote_skips_celsius_markets():
    """The forecast layer requests Fahrenheit and parse_bucket ignores the
    unit, so a Celsius market would be scored against an F forecast and show
    a large fake edge. It must be rejected outright."""
    celsius = _raw(
        question="Will the highest temperature in Seoul be 28°C on July 26?",
        slug="highest-temperature-in-seoul-on-july-26",
    )
    assert build_quote(celsius, TODAY) is None


def test_build_quote_still_accepts_fahrenheit_with_a_city_abbreviation():
    """Guard against the Celsius regex tripping on strings like NYC."""
    quote = build_quote(_raw(), TODAY)
    assert quote is not None and quote.city_key == "nyc"
