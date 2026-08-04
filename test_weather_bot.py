"""Offline tests for weather_bot.py - no network access required.

The sandbox this was written in has outbound network access to
Polymarket/Open-Meteo blocked, so these tests use canned fixtures shaped
like real, live-confirmed API responses instead of hitting the real
services. The public-search event/market shape and the geocoding response
shape below were captured from actual API calls made outside this sandbox.
Run with: python3 -m unittest test_weather_bot.py -v
"""

import unittest
from datetime import date, timedelta
from unittest.mock import patch

import weather_bot as wb


class TestCityParsing(unittest.TestCase):
    def test_extracts_simple_city(self):
        q = "Will the highest temperature in Miami be 100F or higher on August 10?"
        m = wb._CITY_RE.search(q)
        self.assertEqual(m.group(1).strip(), "Miami")

    def test_extracts_multiword_city(self):
        q = "Will the highest temperature in New York City be 92°F or higher on August 3?"
        m = wb._CITY_RE.search(q)
        self.assertEqual(m.group(1).strip(), "New York City")

    def test_extracts_city_with_parenthetical_station(self):
        q = "Will the highest temperature in Seoul (Incheon) be 28°C or below on August 4?"
        m = wb._CITY_RE.search(q)
        self.assertEqual(m.group(1).strip(), "Seoul (Incheon)")


class TestGeocodeCity(unittest.TestCase):
    def setUp(self):
        wb._geocode_cache.clear()

    @patch("weather_bot.requests.get")
    def test_geocodes_simple_city(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "results": [{"latitude": 22.3193, "longitude": 114.1694}]
        }
        coords = wb.geocode_city("Hong Kong")
        self.assertEqual(coords, (22.3193, 114.1694))
        self.assertEqual(mock_get.call_count, 1)

    @patch("weather_bot.requests.get")
    def test_caches_result(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "results": [{"latitude": 51.5074, "longitude": -0.1278}]
        }
        wb.geocode_city("London")
        wb.geocode_city("London")
        self.assertEqual(mock_get.call_count, 1)

    @patch("weather_bot.requests.get")
    def test_falls_back_to_base_name_when_parenthetical_form_fails(self, mock_get):
        # First call (full "Seoul (Incheon)") returns no results, second
        # call (base name "Seoul") succeeds.
        empty = type("R", (), {"raise_for_status": lambda self: None,
                                "json": lambda self: {"results": []}})()
        found = type("R", (), {"raise_for_status": lambda self: None,
                                "json": lambda self: {"results": [{"latitude": 37.5665, "longitude": 126.9780}]}})()
        mock_get.side_effect = [empty, found]
        coords = wb.geocode_city("Seoul (Incheon)")
        self.assertEqual(coords, (37.5665, 126.9780))
        self.assertEqual(mock_get.call_count, 2)

    @patch("weather_bot.requests.get")
    def test_returns_none_and_caches_when_no_candidate_resolves(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {"results": []}
        coords = wb.geocode_city("Nowhereville")
        self.assertIsNone(coords)
        self.assertIn("Nowhereville", wb._geocode_cache)


class TestParsing(unittest.TestCase):
    def setUp(self):
        wb._geocode_cache.clear()

    @patch("weather_bot.geocode_city", return_value=(25.7617, -80.1918))
    def test_parses_fahrenheit_gte(self, _mock_geo):
        q = "Will the highest temperature in Miami be 100F or higher on 2026-08-10?"
        parsed = wb.parse_temperature_market(q)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["city"], "Miami")
        self.assertEqual(parsed["direction"], "gte")
        self.assertAlmostEqual(parsed["threshold_c"], (100 - 32) * 5 / 9, places=3)
        self.assertEqual(parsed["target_date"], date(2026, 8, 10))

    @patch("weather_bot.geocode_city", return_value=(51.5074, -0.1278))
    def test_parses_celsius_lte_with_end_date_fallback(self, _mock_geo):
        q = "Will the highest temperature in London be 25°C or below on August 4?"
        parsed = wb.parse_temperature_market(q, end_date_iso="2026-08-04T12:00:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["city"], "London")
        self.assertEqual(parsed["direction"], "lte")
        self.assertEqual(parsed["threshold_c"], 25.0)
        self.assertEqual(parsed["target_date"], date(2026, 8, 4))

    @patch("weather_bot.geocode_city", return_value=(37.5665, 126.9780))
    def test_parses_city_with_station_parenthetical(self, mock_geo):
        q = "Will the highest temperature in Seoul (Incheon) be 38°C or higher on August 4?"
        parsed = wb.parse_temperature_market(q, end_date_iso="2026-08-04T12:00:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["city"], "Seoul (Incheon)")
        mock_geo.assert_called_once_with("Seoul (Incheon)")

    def test_rejects_non_weather_question(self):
        self.assertIsNone(wb.parse_temperature_market("Will BTC hit $100k?"))

    @patch("weather_bot.geocode_city", return_value=None)
    def test_rejects_ungeocodable_city(self, _mock_geo):
        q = "Will the highest temperature in Atlantis be 100F or higher on 2026-08-10?"
        self.assertIsNone(wb.parse_temperature_market(q))

    def test_rejects_middle_bucket_without_direction(self):
        # The ~9 exact-value buckets per event are intentionally not traded.
        q = "Will the highest temperature in Miami be 95°F on August 10?"
        self.assertIsNone(wb.parse_temperature_market(q))


class TestOutcomeParsing(unittest.TestCase):
    def test_get_outcome_info_from_json_strings(self):
        market = {
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.006", "0.994"]',
            "clobTokenIds": '["tok-yes", "tok-no"]',
        }
        info = wb.get_outcome_info(market)
        self.assertEqual(info["yes_price"], 0.006)
        self.assertEqual(info["yes_token"], "tok-yes")

    def test_get_outcome_info_from_native_lists(self):
        market = {
            "outcomes": ["Yes", "No"],
            "outcomePrices": [0.01, 0.99],
            "clobTokenIds": ["a", "b"],
        }
        info = wb.get_outcome_info(market)
        self.assertEqual(info["yes_price"], 0.01)

    def test_get_outcome_info_missing_fields_returns_none(self):
        self.assertIsNone(wb.get_outcome_info({}))


class TestOrderBook(unittest.TestCase):
    def test_best_bid_ask_dict_levels(self):
        book = {"bids": [{"price": "0.01", "size": "10"}, {"price": "0.03", "size": "5"}],
                "asks": [{"price": "0.05", "size": "10"}, {"price": "0.04", "size": "5"}]}
        self.assertEqual(wb.best_bid(book), 0.03)
        self.assertEqual(wb.best_ask(book), 0.04)

    def test_best_bid_empty(self):
        self.assertIsNone(wb.best_bid({"bids": []}))


class TestProbabilityModel(unittest.TestCase):
    def test_high_confidence_when_models_agree_above_threshold(self):
        temps = [39.0, 39.5, 40.0, 38.8, 39.2, 40.1]
        prob = wb.estimate_probability(temps, threshold_c=32.2, direction="gte")
        self.assertGreater(prob, 0.9)

    def test_low_confidence_when_models_agree_below_threshold(self):
        temps = [25.0, 24.5, 25.5, 24.8]
        prob = wb.estimate_probability(temps, threshold_c=32.2, direction="gte")
        self.assertLess(prob, 0.05)

    def test_none_with_too_few_models(self):
        self.assertIsNone(wb.estimate_probability([25.0], 32.2, "gte"))

    def test_std_floor_prevents_overconfidence(self):
        temps = [32.3, 32.3, 32.3, 32.3]
        prob = wb.estimate_probability(temps, threshold_c=32.2, direction="gte")
        self.assertLess(prob, 0.6)


class TestShouldBuy(unittest.TestCase):
    def test_buys_when_price_in_band_and_edge_large(self):
        self.assertTrue(wb.should_buy(market_price=0.008, model_prob=0.10))

    def test_rejects_price_outside_band(self):
        self.assertFalse(wb.should_buy(market_price=0.05, model_prob=0.30))

    def test_rejects_insufficient_multiple(self):
        self.assertFalse(wb.should_buy(market_price=0.01, model_prob=0.02))

    def test_rejects_none_probability(self):
        self.assertFalse(wb.should_buy(market_price=0.01, model_prob=None))


class TestPaperPortfolio(unittest.TestCase):
    def test_buy_then_take_profit_sell_pnl(self):
        p = wb.PaperPortfolio(starting_cash=100.0)
        ok = p.buy("m1", {"question": "q1", "condition_id": "m1"}, price=0.01, stake=10.0)
        self.assertTrue(ok)
        self.assertEqual(p.cash, 90.0)
        self.assertIn("m1", p.open_positions)

        trade = p.sell("m1", price=0.05, reason="take_profit")
        self.assertIsNotNone(trade)
        self.assertAlmostEqual(p.cash, 90.0 + 10.0 * (0.05 / 0.01), places=6)
        self.assertAlmostEqual(trade["pnl"], 40.0, places=6)
        self.assertNotIn("m1", p.open_positions)
        self.assertEqual(len(p.closed_trades), 1)

    def test_buy_rejects_duplicate_and_insufficient_cash(self):
        p = wb.PaperPortfolio(starting_cash=5.0)
        self.assertFalse(p.buy("m1", {}, price=0.01, stake=10.0))  # not enough cash
        p2 = wb.PaperPortfolio(starting_cash=100.0)
        self.assertTrue(p2.buy("m1", {}, price=0.01, stake=10.0))
        self.assertFalse(p2.buy("m1", {}, price=0.01, stake=10.0))  # duplicate key

    def test_snapshot_win_rate(self):
        p = wb.PaperPortfolio(starting_cash=100.0)
        p.buy("m1", {"question": "q1", "condition_id": "m1"}, price=0.01, stake=10.0)
        p.sell("m1", price=0.02, reason="take_profit")  # win
        p.buy("m2", {"question": "q2", "condition_id": "m2"}, price=0.01, stake=10.0)
        p.sell("m2", price=0.005, reason="resolved")  # loss
        snap = p.snapshot()
        self.assertEqual(snap["total_trades"], 2)
        self.assertEqual(snap["win_rate"], 50.0)


def _bucket_market(condition_id, question, yes_price, closed=False, end_date=None, token="tok"):
    return {
        "conditionId": condition_id,
        "question": question,
        "closed": closed,
        "endDate": end_date,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{yes_price}", "{1 - yes_price}"]',
        "clobTokenIds": f'["{token}-yes", "{token}-no"]',
    }


class TestFetchTemperatureMarkets(unittest.TestCase):
    @patch("weather_bot.requests.get")
    def test_flattens_events_into_markets(self, mock_get):
        target = (date.today() + timedelta(days=1)).isoformat()
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "events": [
                {
                    "title": "Highest temperature in Miami on ...",
                    "markets": [
                        _bucket_market("c1", "Will the highest temperature in Miami be 90F or below ...", 0.01, end_date=f"{target}T00:00:00Z"),
                        _bucket_market("c2", "Will the highest temperature in Miami be 91F ...", 0.3, end_date=f"{target}T00:00:00Z"),
                        _bucket_market("c3", "Will the highest temperature in Miami be 100F or higher ...", 0.005, end_date=f"{target}T00:00:00Z"),
                    ],
                },
                {
                    "title": "Highest temperature in Denver on ...",
                    "markets": [
                        _bucket_market("c4", "Will the highest temperature in Denver be 80F or below ...", 0.02, end_date=f"{target}T00:00:00Z"),
                    ],
                },
            ]
        }
        markets = wb.fetch_temperature_markets()
        self.assertEqual(len(markets), 4)
        self.assertEqual({m["conditionId"] for m in markets}, {"c1", "c2", "c3", "c4"})
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["q"], wb.SEARCH_QUERY)


class TestScanAndManageWithMockedNetwork(unittest.TestCase):
    def setUp(self):
        wb.portfolio = wb.PaperPortfolio(starting_cash=500.0)
        wb._geocode_cache.clear()

    @patch("weather_bot.multi_model_forecast_max")
    @patch("weather_bot.geocode_city", return_value=(25.7617, -80.1918))
    def test_scan_for_entries_buys_on_real_edge(self, _mock_geo, mock_forecast):
        target = (date.today() + timedelta(days=2)).isoformat()
        markets = [_bucket_market(
            "cond-1",
            f"Will the highest temperature in Miami be 95F or higher on {target}?",
            0.008,
            end_date=f"{target}T00:00:00Z",
        )]
        # 95F ~= 35.0C; models clustered well above it -> high model prob.
        mock_forecast.return_value = [38.0, 37.5, 38.2, 37.8]

        wb.scan_for_entries(markets)

        self.assertIn("cond-1", wb.portfolio.open_positions)
        pos = wb.portfolio.open_positions["cond-1"]
        self.assertEqual(pos["city"], "Miami")
        self.assertEqual(wb.portfolio.cash, 500.0 - wb.STAKE_USD)

    @patch("weather_bot.multi_model_forecast_max")
    @patch("weather_bot.geocode_city", return_value=(25.7617, -80.1918))
    def test_scan_for_entries_skips_when_no_edge(self, _mock_geo, mock_forecast):
        target = (date.today() + timedelta(days=2)).isoformat()
        markets = [_bucket_market(
            "cond-2",
            f"Will the highest temperature in Miami be 95F or higher on {target}?",
            0.008,
            end_date=f"{target}T00:00:00Z",
        )]
        mock_forecast.return_value = [20.0, 19.5, 20.2, 19.8]

        wb.scan_for_entries(markets)

        self.assertNotIn("cond-2", wb.portfolio.open_positions)
        self.assertEqual(wb.portfolio.cash, 500.0)

    @patch("weather_bot.geocode_city", return_value=(25.7617, -80.1918))
    def test_scan_for_entries_skips_closed_markets(self, _mock_geo):
        target = (date.today() + timedelta(days=2)).isoformat()
        markets = [_bucket_market(
            "cond-5",
            f"Will the highest temperature in Miami be 95F or higher on {target}?",
            0.0005,
            closed=True,
            end_date=f"{target}T00:00:00Z",
        )]
        wb.scan_for_entries(markets)
        self.assertNotIn("cond-5", wb.portfolio.open_positions)

    @patch("weather_bot.get_order_book")
    def test_manage_open_positions_take_profit(self, mock_book):
        wb.portfolio.buy("cond-3", {"question": "q", "condition_id": "cond-3"}, price=0.01, stake=10.0)
        markets = [_bucket_market("cond-3", "q", 0.05, token="tok3")]
        mock_book.return_value = {"bids": [{"price": "0.04", "size": "100"}], "asks": []}

        wb.manage_open_positions(markets)

        self.assertNotIn("cond-3", wb.portfolio.open_positions)
        self.assertEqual(len(wb.portfolio.closed_trades), 1)
        self.assertEqual(wb.portfolio.closed_trades[0]["reason"], "take_profit")

    def test_manage_open_positions_settles_resolved_market(self):
        wb.portfolio.buy("cond-4", {"question": "q", "condition_id": "cond-4"}, price=0.01, stake=10.0)
        markets = [_bucket_market("cond-4", "q", 1.0, closed=True)]

        wb.manage_open_positions(markets)

        self.assertNotIn("cond-4", wb.portfolio.open_positions)
        trade = wb.portfolio.closed_trades[0]
        self.assertEqual(trade["reason"], "resolved")
        self.assertAlmostEqual(trade["pnl"], 990.0, places=6)  # 1000 shares * $1 - $10 stake

    @patch("weather_bot.fetch_market_by_condition_id")
    def test_manage_open_positions_falls_back_when_not_in_fresh_markets(self, mock_fetch):
        wb.portfolio.buy("cond-6", {"question": "q", "condition_id": "cond-6"}, price=0.01, stake=10.0)
        mock_fetch.return_value = _bucket_market("cond-6", "q", 1.0, closed=True)

        wb.manage_open_positions(markets=[])  # position not present in this cycle's search results

        mock_fetch.assert_called_once_with("cond-6")
        self.assertNotIn("cond-6", wb.portfolio.open_positions)


if __name__ == "__main__":
    unittest.main()
