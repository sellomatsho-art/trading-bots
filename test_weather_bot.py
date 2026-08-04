"""Offline tests for weather_bot.py - no network access required.

The sandbox this was written in has outbound network access to
Polymarket/Open-Meteo blocked, so these tests use canned fixtures shaped
like the documented API responses instead of hitting the real services.
Run with: python3 -m unittest test_weather_bot.py -v
"""

import unittest
from datetime import date, timedelta
from unittest.mock import patch

import weather_bot as wb


class TestParsing(unittest.TestCase):
    def test_parses_fahrenheit_gte(self):
        q = "Will the highest temperature in NYC be 100F or higher on 2026-08-10?"
        parsed = wb.parse_temperature_market(q)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["city"], "New York City")
        self.assertEqual(parsed["direction"], "gte")
        self.assertAlmostEqual(parsed["threshold_c"], (100 - 32) * 5 / 9, places=3)
        self.assertEqual(parsed["target_date"], date(2026, 8, 10))

    def test_parses_celsius_lte_with_end_date_fallback(self):
        q = "Record low: highest temperature in London 5C or lower?"
        parsed = wb.parse_temperature_market(q, end_date_iso="2026-01-15T00:00:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["city"], "London")
        self.assertEqual(parsed["direction"], "lte")
        self.assertEqual(parsed["threshold_c"], 5.0)
        self.assertEqual(parsed["target_date"], date(2026, 1, 15))

    def test_rejects_non_weather_question(self):
        self.assertIsNone(wb.parse_temperature_market("Will BTC hit $100k?"))

    def test_rejects_unknown_city(self):
        q = "Highest temperature in Atlantis 100F or higher on 2026-08-10?"
        self.assertIsNone(wb.parse_temperature_market(q))

    def test_rejects_missing_direction(self):
        q = "Highest temperature in Miami 100F on 2026-08-10?"
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
        # Models agreeing to the decimal shouldn't imply near-certainty this close to the line.
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


class TestScanAndManageWithMockedNetwork(unittest.TestCase):
    def setUp(self):
        wb.portfolio = wb.PaperPortfolio(starting_cash=500.0)

    @patch("weather_bot.multi_model_forecast_max")
    @patch("weather_bot.fetch_temperature_markets")
    def test_scan_for_entries_buys_on_real_edge(self, mock_fetch, mock_forecast):
        target = (date.today() + timedelta(days=2)).isoformat()
        mock_fetch.return_value = [{
            "conditionId": "cond-1",
            "question": f"Highest temperature in Miami 95F or higher on {target}?",
            "endDate": f"{target}T00:00:00Z",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.008", "0.992"]',
            "clobTokenIds": '["tok-yes-1", "tok-no-1"]',
        }]
        # 95F ~= 35.0C; models clustered well above it -> high model prob.
        mock_forecast.return_value = [38.0, 37.5, 38.2, 37.8]

        wb.scan_for_entries()

        self.assertIn("cond-1", wb.portfolio.open_positions)
        pos = wb.portfolio.open_positions["cond-1"]
        self.assertEqual(pos["city"], "Miami")
        self.assertEqual(wb.portfolio.cash, 500.0 - wb.STAKE_USD)

    @patch("weather_bot.multi_model_forecast_max")
    @patch("weather_bot.fetch_temperature_markets")
    def test_scan_for_entries_skips_when_no_edge(self, mock_fetch, mock_forecast):
        target = (date.today() + timedelta(days=2)).isoformat()
        mock_fetch.return_value = [{
            "conditionId": "cond-2",
            "question": f"Highest temperature in Miami 95F or higher on {target}?",
            "endDate": f"{target}T00:00:00Z",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.008", "0.992"]',
            "clobTokenIds": '["tok-yes-2", "tok-no-2"]',
        }]
        # Models clustered well below threshold -> low model prob, no edge.
        mock_forecast.return_value = [20.0, 19.5, 20.2, 19.8]

        wb.scan_for_entries()

        self.assertNotIn("cond-2", wb.portfolio.open_positions)
        self.assertEqual(wb.portfolio.cash, 500.0)

    @patch("weather_bot.get_order_book")
    @patch("weather_bot.fetch_market_by_condition_id")
    def test_manage_open_positions_take_profit(self, mock_fetch_market, mock_book):
        wb.portfolio.buy("cond-3", {
            "question": "q", "condition_id": "cond-3",
        }, price=0.01, stake=10.0)
        mock_fetch_market.return_value = {
            "closed": False,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.05", "0.95"]',
            "clobTokenIds": '["tok-yes-3", "tok-no-3"]',
        }
        mock_book.return_value = {"bids": [{"price": "0.04", "size": "100"}], "asks": []}

        wb.manage_open_positions()

        self.assertNotIn("cond-3", wb.portfolio.open_positions)
        self.assertEqual(len(wb.portfolio.closed_trades), 1)
        self.assertEqual(wb.portfolio.closed_trades[0]["reason"], "take_profit")

    @patch("weather_bot.fetch_market_by_condition_id")
    def test_manage_open_positions_settles_resolved_market(self, mock_fetch_market):
        wb.portfolio.buy("cond-4", {
            "question": "q", "condition_id": "cond-4",
        }, price=0.01, stake=10.0)
        mock_fetch_market.return_value = {
            "closed": True,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1.0", "0.0"]',
        }

        wb.manage_open_positions()

        self.assertNotIn("cond-4", wb.portfolio.open_positions)
        trade = wb.portfolio.closed_trades[0]
        self.assertEqual(trade["reason"], "resolved")
        self.assertAlmostEqual(trade["pnl"], 990.0, places=6)  # 1000 shares * $1 - $10 stake


if __name__ == "__main__":
    unittest.main()
