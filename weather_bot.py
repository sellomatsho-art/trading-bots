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

Market discovery uses Gamma's public-search endpoint (confirmed live):
querying "highest temperature" returns the active per-city "Highest
temperature in <City> on <date>?" events, each containing ~11 sub-markets
(one per temperature bucket). Only the two extreme buckets are phrased
with "or higher"/"or below" - the ones parse_temperature_market() accepts
- so only the tail contracts this strategy targets get considered; the
~9 middle exact-value buckets are ignored by design. City coordinates are
resolved dynamically via Open-Meteo's geocoding API rather than a fixed
list, since the set of cities Polymarket covers changes and is larger
than any reasonable hardcoded list (confirmed live: Hong Kong, Seoul
(Incheon), Chongqing, Chengdu, Busan, Wellington and more, none of which
would be in a short guessed list).
"""

import csv
import json
import math
import os
import re
import threading
import time
from collections import deque
from datetime import date, datetime

import requests

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
PUBLIC_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless,meteofrance_seamless,jma_seamless"
SEARCH_QUERY = "highest temperature"
SEARCH_LIMIT = 50

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

_THRESHOLD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°?\s*(F|C)\b", re.I)
_DIRECTION_GTE_RE = re.compile(r"\bor\s+(higher|above|more|greater)\b", re.I)
_DIRECTION_LTE_RE = re.compile(r"\bor\s+(lower|below|less)\b", re.I)
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# Matches "highest temperature in <City> be" / "... in <City> on" - covers
# both "Will the highest temperature in Seoul (Incheon) be 28C ..." and any
# phrasing that goes straight from city to date without "be".
_CITY_RE = re.compile(r"highest temperature in (.+?)\s+(?:be|on)\b", re.I)

_geocode_cache = {}


def geocode_city(name):
    """Resolve a city name to (lat, lon) via Open-Meteo's geocoding API,
    cached in-memory. Handles "City (Station)" forms like "Seoul (Incheon)"
    by trying the parenthetical first (that's the actual measurement
    station these markets resolve against - e.g. "Seoul (Incheon)" means
    the Incheon station, which can read meaningfully differently than
    downtown Seoul), then the base city, then the full string."""
    if name in _geocode_cache:
        return _geocode_cache[name]

    candidates = []
    paren = re.search(r"\(([^)]+)\)", name)
    if paren:
        candidates.append(paren.group(1).strip())
    base = re.sub(r"\s*\([^)]*\)\s*", "", name).strip()
    if base and base not in candidates:
        candidates.append(base)
    if name not in candidates:
        candidates.append(name)

    coords = None
    for candidate in candidates:
        try:
            r = requests.get(GEOCODE_URL, params={"name": candidate, "count": 1}, timeout=10)
            r.raise_for_status()
            results = r.json().get("results")
            if results:
                coords = (results[0]["latitude"], results[0]["longitude"])
                break
        except Exception as e:
            log_error(f"geocode_city[{candidate}]: {e}")

    _geocode_cache[name] = coords
    return coords


def _fahrenheit_to_celsius(f):
    return (f - 32.0) * 5.0 / 9.0


def parse_temperature_market(question, end_date_iso=None):
    """Best-effort parse of a Polymarket extreme-temperature-record question.

    Only matches the extreme tail buckets ("... or higher" / "... or
    below") - the ~9 middle exact-value buckets in each event are
    intentionally ignored, matching the strategy's focus on cheap tail
    contracts rather than the likely-priced middle of the distribution.

    Returns a dict with city/coords/threshold_c/direction/target_date, or
    None if the question doesn't look like this kind of market.
    """
    if "temperature" not in question.lower():
        return None

    city_m = _CITY_RE.search(question)
    if not city_m:
        return None
    city = city_m.group(1).strip()

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

    coords = geocode_city(city)
    if coords is None:
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


def fetch_temperature_markets():
    """The flat /markets list is huge (thousands of active markets, arbitrary
    order) and paging through it misses these entirely - public-search finds
    them directly by matching on the event title/description. Returns a flat
    list of individual sub-markets (temperature buckets) across all matching
    events."""
    r = requests.get(PUBLIC_SEARCH_URL, params={
        "q": SEARCH_QUERY, "limit_per_type": SEARCH_LIMIT,
    }, timeout=15)
    r.raise_for_status()
    events = r.json().get("events", [])
    markets = []
    for e in events:
        markets.extend(e.get("markets", []))
    return markets


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


def manage_open_positions(markets=None):
    """markets: optionally the list already fetched this cycle by
    scan_for_entries, keyed by conditionId - avoids a second full search
    call. Falls back to a direct per-position lookup if a position's
    market isn't in that list (e.g. it aged out of search results)."""
    if not portfolio.open_positions:
        return
    by_condition_id = {m.get("conditionId"): m for m in (markets or []) if m.get("conditionId")}
    for key, pos in list(portfolio.open_positions.items()):
        try:
            m = by_condition_id.get(pos["condition_id"])
            if m is None:
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


def scan_for_entries(markets):
    if len(portfolio.open_positions) >= MAX_OPEN_POSITIONS:
        return

    for m in markets:
        if len(portfolio.open_positions) >= MAX_OPEN_POSITIONS:
            break
        if m.get("closed"):
            continue
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


# ---------------------------------------------------------------------------
# Shadow logging - records what the model PREDICTED, independently of what the
# bot traded, so the model can be scored before anything is staked on it.
#
# Why this exists. should_buy() demands model_prob >= 4x the market price, and
# at these prices that puts the decision 2-4 sigma into the tail, where the
# number is a property of the distribution assumption rather than of the
# forecast. Measured 4 Aug 2026 on a live Seattle contract: the raw 6-model
# spread was 0.64C, MIN_STD_C floors it to 1.2, and that single substitution
# moves the probability from 0.00008 to 0.02225 - a 268x change with no change
# in the weather. Nudging the floor to 1.5 flips that market from no-trade to
# trade. Until there is a calibration record, tuning that constant only picks
# how often the bot fires, not whether it is right.
#
# Deliberately NOT wired into scan_for_entries(): trading behaviour must stay
# byte-identical, so this is a separate pass that only reads and appends.
#
# Two design choices that matter:
#
#   1. The raw per-model temperatures are logged, not just the derived
#      probability. Any sigma floor, lead-time sigma model or non-normal
#      distribution can then be re-scored against history WITHOUT re-collecting
#      six weeks of forecasts. The probability is a derived column; the
#      ensemble is the actual observation.
#   2. One row per (market, days_ahead), not one per poll. POLL_SECONDS=300
#      would otherwise write ~288 near-identical rows per market per day and
#      swamp any calibration with duplicates. Keying on lead time still lets a
#      day-6 call and a day-0 call on the same market both be scored, which is
#      the interesting comparison.
#
# Dedupe happens BEFORE the forecast call, so a cycle with nothing new to say
# costs zero Open-Meteo requests. Only the price band is dropped relative to
# scan_for_entries - the model is evaluated across the whole horizon window
# (~56 markets vs the ~4 that clear MIN_PRICE), which is what makes the record
# accumulate at a usable rate.
SHADOW_CSV = "weather_shadow.csv"
SHADOW_FIELDS = [
    "logged_at", "condition_id", "question", "city", "target_date",
    "days_ahead", "threshold_c", "direction", "market_price",
    "model_prob", "would_buy", "n_models", "ensemble_temps_c",
    "ensemble_mean_c", "ensemble_std_raw_c", "sigma_used_c", "min_std_c",
    "resolved_at", "outcome", "resolve_price",
]
_shadow_seen = set()


def _shadow_load_seen():
    """(condition_id, days_ahead) pairs already written, so a restart does not
    duplicate rows. Read once at import; the file is append-only afterwards."""
    if not os.path.exists(SHADOW_CSV):
        return
    try:
        with open(SHADOW_CSV, newline="") as f:
            for row in csv.DictReader(f):
                _shadow_seen.add((row.get("condition_id"), row.get("days_ahead")))
    except Exception as e:
        log_error(f"_shadow_load_seen: {e}")


def _shadow_append(row):
    new = not os.path.exists(SHADOW_CSV)
    with open(SHADOW_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHADOW_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def shadow_scan(markets):
    """Score every in-horizon market and append one row per new
    (market, days_ahead). Reads only; never buys, never touches portfolio."""
    forecasts = {}
    written = 0
    for m in markets:
        try:
            if m.get("closed"):
                continue
            cid = m.get("conditionId") or m.get("id")
            if not cid:
                continue
            parsed = parse_temperature_market(m.get("question", ""), m.get("endDate"))
            if not parsed:
                continue
            days_ahead = (parsed["target_date"] - date.today()).days
            if not (0 <= days_ahead <= MAX_DAYS_AHEAD):
                continue
            key = (cid, str(days_ahead))
            if key in _shadow_seen:
                continue            # dedupe BEFORE any network call
            info = get_outcome_info(m)
            if not info:
                continue
            fkey = (round(parsed["lat"], 3), round(parsed["lon"], 3),
                    parsed["target_date"].isoformat())
            if fkey not in forecasts:
                forecasts[fkey] = multi_model_forecast_max(*fkey)
            temps = forecasts[fkey]
            if not temps or len(temps) < 2:
                continue
            mean = sum(temps) / len(temps)
            var = sum((t - mean) ** 2 for t in temps) / (len(temps) - 1)
            raw_std = var ** 0.5
            prob = estimate_probability(temps, parsed["threshold_c"], parsed["direction"])
            price = info["yes_price"]
            _shadow_append({
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "condition_id": cid,
                "question": m.get("question", ""),
                "city": parsed["city"],
                "target_date": parsed["target_date"].isoformat(),
                "days_ahead": days_ahead,
                "threshold_c": round(parsed["threshold_c"], 3),
                "direction": parsed["direction"],
                "market_price": price,
                "model_prob": None if prob is None else round(prob, 6),
                "would_buy": should_buy(price, prob),
                "n_models": len(temps),
                "ensemble_temps_c": json.dumps([round(t, 2) for t in temps]),
                "ensemble_mean_c": round(mean, 3),
                "ensemble_std_raw_c": round(raw_std, 3),
                "sigma_used_c": round(max(raw_std, MIN_STD_C), 3),
                "min_std_c": MIN_STD_C,
                "resolved_at": "", "outcome": "", "resolve_price": "",
            })
            _shadow_seen.add(key)
            written += 1
        except Exception as e:
            log_error(f"shadow_scan[{m.get('conditionId')}]: {e}")
    return written


def resolve_shadow_rows(limit=25):
    """Fill in the outcome for logged predictions whose market has since
    closed. Rewrites the file in place, so it is capped per pass and only runs
    when there is something whose target date has already passed."""
    if not os.path.exists(SHADOW_CSV):
        return 0
    try:
        with open(SHADOW_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        log_error(f"resolve_shadow_rows read: {e}")
        return 0

    today = date.today().isoformat()
    pending = [r for r in rows
               if not r.get("outcome") and r.get("target_date", "9999") < today]
    if not pending:
        return 0

    resolved = {}
    for r in pending[:limit]:
        cid = r["condition_id"]
        if cid in resolved:
            continue
        try:
            m = fetch_market_by_condition_id(cid)
            if not m or not m.get("closed"):
                continue
            info = get_outcome_info(m)
            if not info:
                continue
            yes = info["yes_price"]
            resolved[cid] = ("YES" if yes > 0.5 else "NO", yes)
        except Exception as e:
            log_error(f"resolve_shadow_rows[{cid}]: {e}")

    if not resolved:
        return 0
    stamp = datetime.now().isoformat(timespec="seconds")
    n = 0
    for r in rows:
        if not r.get("outcome") and r["condition_id"] in resolved:
            outcome, price = resolved[r["condition_id"]]
            r["outcome"], r["resolve_price"], r["resolved_at"] = outcome, price, stamp
            n += 1
    try:
        tmp = SHADOW_CSV + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHADOW_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, SHADOW_CSV)
    except Exception as e:
        log_error(f"resolve_shadow_rows write: {e}")
        return 0
    return n


_shadow_load_seen()


def weather_strategy_loop():
    while True:
        markets = []
        try:
            markets = fetch_temperature_markets()
        except Exception as e:
            log_error(f"fetch_temperature_markets: {e}")
        try:
            manage_open_positions(markets)
            scan_for_entries(markets)
        except Exception as e:
            log_error(f"weather_strategy_loop: {e}")
        # Separate try: shadow logging is observational, and a failure in it
        # must never stop the bot from trading or managing open positions.
        try:
            shadow_scan(markets)
            resolve_shadow_rows()
        except Exception as e:
            log_error(f"shadow logging: {e}")
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
