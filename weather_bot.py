"""
Paper-trading strategy for Polymarket extreme-temperature-record contracts.

Idea being tested: markets like "Will the highest temperature in <City> be
N or higher/lower on <date>?" sometimes trade at 0.3-1.5c (implying a
<1.5% probability) while a short-range multi-model weather forecast puts
the real probability meaningfully higher. This module scans for that gap,
paper-buys the cheap side when the gap is large enough, and paper-sells
early once the market price runs up (rather than holding to resolution).

NO REAL ORDERS ARE EVER PLACED. Everything here is simulated bookkeeping
against live market prices/forecasts, tracked in PaperPortfolio.

Network note: Polymarket's Gamma/CLOB APIs and Open-Meteo were not
reachable from the sandbox this was built in (outbound network policy),
so the exact JSON field names below are based on documented API shapes
and could not be live-verified. fetch_active_markets(),
get_outcome_info(), get_order_book() and multi_model_forecast_max() are
the functions to double check first against a live response if trades
aren't showing up as expected.
"""

import json
import math
import re
import threading
import time
from collections import deque
from datetime import date, datetime

import requests

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless,meteofrance_seamless,jma_seamless"

# Only buy inside roughly the price band the strategy this is modeled on used.
MIN_PRICE = 0.003
MAX_PRICE = 0.015
# Require the forecast-implied probability to be both a multiple of, and
# meaningfully larger in absolute terms than, the market price - avoids
# firing on noise when the market price is a fraction of a cent.
EDGE_MULTIPLE = 4.0
MIN_EDGE_ABS = 0.015
# Sell early once price has run up this much, instead of holding to resolution.
TAKE_PROFIT_MULTIPLE = 3.0
STAKE_USD = 10.0
MAX_OPEN_POSITIONS = 8
POLL_SECONDS = 300
# Multi-model NWP forecasts stop being informative for extremes much beyond
# this horizon, so don't evaluate markets resolving further out than this.
MAX_DAYS_AHEAD = 6
# Floor on the model-temperature std dev so a handful of models happening to
# agree closely doesn't produce an overconfident probability.
MIN_STD_C = 1.2

CITY_ALIASES = {
    "new york city": "New York City", "new york": "New York City", "nyc": "New York City",
    "los angeles": "Los Angeles",
    "chicago": "Chicago",
    "miami": "Miami",
    "houston": "Houston",
    "phoenix": "Phoenix",
    "philadelphia": "Philadelphia",
    "austin": "Austin",
    "dallas": "Dallas",
    "san antonio": "San Antonio",
    "san diego": "San Diego",
    "denver": "Denver",
    "seattle": "Seattle",
    "boston": "Boston",
    "atlanta": "Atlanta",
    "las vegas": "Las Vegas",
    "washington dc": "Washington DC", "washington d.c.": "Washington DC",
    "london": "London",
    "paris": "Paris",
    "moscow": "Moscow",
    "tokyo": "Tokyo",
    "sydney": "Sydney",
}

CITY_COORDS = {
    "New York City": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Miami": (25.7617, -80.1918),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "Austin": (30.2672, -97.7431),
    "Dallas": (32.7767, -96.7970),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Denver": (39.7392, -104.9903),
    "Seattle": (47.6062, -122.3321),
    "Boston": (42.3601, -71.0589),
    "Atlanta": (33.7490, -84.3880),
    "Las Vegas": (36.1699, -115.1398),
    "Washington DC": (38.9072, -77.0369),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Moscow": (55.7558, 37.6173),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8688, 151.2093),
}

_ALIASES_BY_LENGTH = sorted(CITY_ALIASES.items(), key=lambda kv: -len(kv[0]))

_THRESHOLD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°?\s*(F|C)\b", re.I)
_DIRECTION_GTE_RE = re.compile(r"\bor\s+(higher|above|more|greater)\b", re.I)
_DIRECTION_LTE_RE = re.compile(r"\bor\s+(lower|below|less)\b", re.I)
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def find_city(question):
    ql = question.lower()
    for alias, canonical in _ALIASES_BY_LENGTH:
        if re.search(r"\b" + re.escape(alias) + r"\b", ql):
            return canonical, CITY_COORDS[canonical]
    return None, None


def _fahrenheit_to_celsius(f):
    return (f - 32.0) * 5.0 / 9.0


def parse_temperature_market(question, end_date_iso=None):
    """Best-effort parse of a Polymarket extreme-temperature-record question.

    Returns a dict with city/coords/threshold_c/direction/target_date, or
    None if the question doesn't look like this kind of market.
    """
    ql = question.lower()
    if "temperature" not in ql:
        return None
    if not any(k in ql for k in ("highest", "record", "high ")):
        return None

    city, coords = find_city(question)
    if not city:
        return None

    m = _THRESHOLD_RE.search(question)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).upper()
    threshold_c = value if unit == "C" else _fahrenheit_to_celsius(value)

    if _DIRECTION_GTE_RE.search(question):
        direction = "gte"
    elif _DIRECTION_LTE_RE.search(question):
        direction = "lte"
    else:
        return None

    target_date = None
    dm = _DATE_RE.search(question)
    if dm:
        target_date = date(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
    elif end_date_iso:
        try:
            target_date = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00")).date()
        except ValueError:
            target_date = None
    if target_date is None:
        return None

    return {
        "city": city,
        "lat": coords[0],
        "lon": coords[1],
        "threshold_c": threshold_c,
        "direction": direction,
        "target_date": target_date,
    }


def _as_list(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value


def get_outcome_info(market):
    """Extract the YES side's current price and CLOB token id from a Gamma
    market object. Returns None if the market doesn't have a usable YES
    outcome (shape mismatch, or a non-binary market)."""
    outcomes = _as_list(market.get("outcomes"))
    prices = _as_list(market.get("outcomePrices"))
    tokens = _as_list(market.get("clobTokenIds"))
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    for i, name in enumerate(outcomes):
        if str(name).strip().lower() == "yes":
            try:
                price = float(prices[i])
            except (TypeError, ValueError):
                return None
            token = tokens[i] if tokens and i < len(tokens) else None
            return {"yes_price": price, "yes_token": token}
    return None


def _price_of(level):
    if isinstance(level, dict):
        return float(level.get("price"))
    return float(level[0])


def get_order_book(token_id):
    r = requests.get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def best_bid(book):
    bids = book.get("bids") or []
    if not bids:
        return None
    return max(_price_of(b) for b in bids)


def best_ask(book):
    asks = book.get("asks") or []
    if not asks:
        return None
    return min(_price_of(a) for a in asks)


def fetch_active_markets(limit=100, max_pages=5):
    out = []
    offset = 0
    for _ in range(max_pages):
        r = requests.get(GAMMA_MARKETS_URL, params={
            "active": "true", "closed": "false", "limit": limit, "offset": offset,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return out


def fetch_temperature_markets():
    return [
        m for m in fetch_active_markets()
        if "temperature" in m.get("question", "").lower()
    ]


def fetch_market_by_condition_id(condition_id):
    r = requests.get(GAMMA_MARKETS_URL, params={"condition_ids": condition_id}, timeout=15)
    r.raise_for_status()
    batch = r.json()
    return batch[0] if batch else None


def multi_model_forecast_max(lat, lon, target_date_iso):
    """Daily max temperature (Celsius) forecast for target_date from each of
    several independent NWP models - a cheap stand-in for a full ensemble."""
    r = requests.get(OPEN_METEO_URL, params={
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "celsius",
        "models": FORECAST_MODELS,
        "timezone": "auto",
        "start_date": target_date_iso,
        "end_date": target_date_iso,
    }, timeout=15)
    r.raise_for_status()
    daily = r.json().get("daily", {})
    dates = daily.get("time", [])
    if target_date_iso not in dates:
        return []
    idx = dates.index(target_date_iso)
    temps = []
    for key, values in daily.items():
        if key == "time":
            continue
        if idx < len(values) and values[idx] is not None:
            temps.append(float(values[idx]))
    return temps


def _normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def estimate_probability(temps_c, threshold_c, direction, min_std=MIN_STD_C):
    """Probability the daily max crosses threshold_c, from the spread of
    per-model forecasts (mean/std -> normal CDF), floored so a handful of
    models agreeing closely doesn't read as certainty."""
    if not temps_c or len(temps_c) < 2:
        return None
    mean = sum(temps_c) / len(temps_c)
    var = sum((t - mean) ** 2 for t in temps_c) / (len(temps_c) - 1)
    std = max(var ** 0.5, min_std)
    if direction == "gte":
        z = (mean - threshold_c) / std
    else:
        z = (threshold_c - mean) / std
    return _normal_cdf(z)


def should_buy(market_price, model_prob):
    if model_prob is None:
        return False
    if not (MIN_PRICE <= market_price <= MAX_PRICE):
        return False
    if model_prob < market_price * EDGE_MULTIPLE:
        return False
    if (model_prob - market_price) < MIN_EDGE_ABS:
        return False
    return True


class PaperPortfolio:
    """In-memory simulated portfolio. No real orders, no real funds."""

    def __init__(self, starting_cash=500.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.open_positions = {}
        self.closed_trades = []
        self.lock = threading.Lock()

    def buy(self, key, info, price, stake):
        with self.lock:
            if key in self.open_positions or stake > self.cash or price <= 0:
                return False
            shares = stake / price
            self.cash -= stake
            self.open_positions[key] = {
                **info,
                "entry_price": price,
                "shares": shares,
                "stake": stake,
                "entry_time": datetime.now().isoformat(timespec="seconds"),
            }
            return True

    def sell(self, key, price, reason):
        with self.lock:
            pos = self.open_positions.pop(key, None)
            if not pos:
                return None
            proceeds = pos["shares"] * price
            pnl = proceeds - pos["stake"]
            self.cash += proceeds
            trade = {
                **pos,
                "exit_price": price,
                "exit_time": datetime.now().isoformat(timespec="seconds"),
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_pct": (pnl / pos["stake"]) * 100 if pos["stake"] else 0,
                "reason": reason,
            }
            self.closed_trades.append(trade)
            return trade

    def snapshot(self):
        with self.lock:
            realized_pnl = sum(t["pnl"] for t in self.closed_trades)
            wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
            total = len(self.closed_trades)
            return {
                "cash": round(self.cash, 2),
                "starting_cash": self.starting_cash,
                "realized_pnl": round(realized_pnl, 2),
                "open_positions": list(self.open_positions.values()),
                "closed_trades": list(reversed(self.closed_trades)),
                "win_rate": round(100 * wins / total, 1) if total else None,
                "total_trades": total,
            }


portfolio = PaperPortfolio(starting_cash=500.0)
_recent_errors = deque(maxlen=20)


def log_error(msg):
    _recent_errors.append(f"{datetime.now().isoformat(timespec='seconds')} {msg}")


def manage_open_positions():
    for key, pos in list(portfolio.open_positions.items()):
        try:
            m = fetch_market_by_condition_id(pos["condition_id"])
            if m is None:
                continue
            if m.get("closed"):
                info = get_outcome_info(m)
                if info is not None:
                    portfolio.sell(key, info["yes_price"], reason="resolved")
                continue
            info = get_outcome_info(m)
            if not info or not info.get("yes_token"):
                continue
            book = get_order_book(info["yes_token"])
            bid = best_bid(book)
            if bid is not None and bid >= pos["entry_price"] * TAKE_PROFIT_MULTIPLE:
                portfolio.sell(key, bid, reason="take_profit")
        except Exception as e:
            log_error(f"manage_open_positions[{key}]: {e}")


def scan_for_entries():
    if len(portfolio.open_positions) >= MAX_OPEN_POSITIONS:
        return
    try:
        markets = fetch_temperature_markets()
    except Exception as e:
        log_error(f"fetch_temperature_markets: {e}")
        return

    for m in markets:
        if len(portfolio.open_positions) >= MAX_OPEN_POSITIONS:
            break
        key = m.get("conditionId") or m.get("id")
        if not key or key in portfolio.open_positions:
            continue
        parsed = parse_temperature_market(m.get("question", ""), m.get("endDate"))
        if not parsed:
            continue
        days_ahead = (parsed["target_date"] - date.today()).days
        if not (0 <= days_ahead <= MAX_DAYS_AHEAD):
            continue
        info = get_outcome_info(m)
        if not info or not info.get("yes_token"):
            continue
        price = info["yes_price"]
        if not (MIN_PRICE <= price <= MAX_PRICE):
            continue
        try:
            temps = multi_model_forecast_max(parsed["lat"], parsed["lon"], parsed["target_date"].isoformat())
        except Exception as e:
            log_error(f"multi_model_forecast_max[{key}]: {e}")
            continue
        model_prob = estimate_probability(temps, parsed["threshold_c"], parsed["direction"])
        if not should_buy(price, model_prob):
            continue
        portfolio.buy(key, {
            "question": m.get("question"),
            "condition_id": key,
            "city": parsed["city"],
            "threshold_c": round(parsed["threshold_c"], 1),
            "direction": parsed["direction"],
            "target_date": parsed["target_date"].isoformat(),
            "model_prob": round(model_prob, 4),
            "yes_token": info["yes_token"],
        }, price, STAKE_USD)


def weather_strategy_loop():
    while True:
        try:
            manage_open_positions()
            scan_for_entries()
        except Exception as e:
            log_error(f"weather_strategy_loop: {e}")
        time.sleep(POLL_SECONDS)


def _fmt_usd(x):
    return f"${x:,.2f}"


def render_status_html():
    snap = portfolio.snapshot()
    rows_open = "".join(
        f"<tr><td>{p['question']}</td><td>{_fmt_usd(p['entry_price']*100)}c</td>"
        f"<td>{_fmt_usd(p['stake'])}</td><td>{p.get('model_prob', '')}</td>"
        f"<td>{p['entry_time']}</td></tr>"
        for p in snap["open_positions"]
    ) or "<tr><td colspan=5>none</td></tr>"

    rows_closed = "".join(
        f"<tr><td>{t['question']}</td><td>{_fmt_usd(t['entry_price']*100)}c -> {_fmt_usd(t['exit_price']*100)}c</td>"
        f"<td>{_fmt_usd(t['pnl'])} ({t['pnl_pct']:.0f}%)</td><td>{t['reason']}</td></tr>"
        for t in snap["closed_trades"]
    ) or "<tr><td colspan=4>none</td></tr>"

    errors = "".join(f"<li>{e}</li>" for e in list(_recent_errors)[-5:]) or "<li>none</li>"

    return f"""
    <h1>Weather Contract Bot - PAPER TRADING (no real funds)</h1>
    <p>Cash: {_fmt_usd(snap['cash'])} / Starting: {_fmt_usd(snap['starting_cash'])}</p>
    <p>Realized P&amp;L: {_fmt_usd(snap['realized_pnl'])} | Trades: {snap['total_trades']} | Win rate: {snap['win_rate']}</p>
    <h2>Open positions</h2>
    <table border=1 cellpadding=4><tr><th>Market</th><th>Entry</th><th>Stake</th><th>Model P</th><th>Since</th></tr>{rows_open}</table>
    <h2>Closed trades</h2>
    <table border=1 cellpadding=4><tr><th>Market</th><th>Entry -> Exit</th><th>P&amp;L</th><th>Reason</th></tr>{rows_closed}</table>
    <h2>Recent errors</h2>
    <ul>{errors}</ul>
    """
