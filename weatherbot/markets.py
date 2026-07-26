"""Read-only Polymarket market data via the public Gamma API.

This module only ever performs HTTP GETs against a public endpoint. There is no
order placement, no CLOB client, no signer and no wallet -- by design.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

from .cities import City, find_city

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
TIMEOUT = 30
PAGE_SIZE = 500

#: A market is a temperature market only if it says so.
TEMPERATURE_RE = re.compile(r"\b(temperature|temp|degrees|high(?:est)? temp)\b", re.IGNORECASE)

_DEG = r"(?:\s*°\s*F?|\s*(?:deg(?:rees)?|F)\b)?"
_NUM = r"(-?\d{1,3})"

_RANGE_RE = re.compile(rf"{_NUM}{_DEG}\s*(?:-|–|—|to)\s*{_NUM}{_DEG}", re.IGNORECASE)

# Inclusive and exclusive bounds are kept apart on purpose. "90 or above"
# includes 90; "above 90" does not. Collapsing the two shifts every threshold
# market by one degree, which on a 1-2 degree bucket is the whole edge.
_ABOVE_INCL_RE = re.compile(
    rf"(?:at least\s*{_NUM}{_DEG})"
    rf"|(?:{_NUM}{_DEG}\s*(?:\+|or (?:above|higher|more|greater|hotter|warmer)))",
    re.IGNORECASE,
)
_ABOVE_EXCL_RE = re.compile(
    rf"(?:above|over|greater than|higher than|more than|hotter than|warmer than)\s*{_NUM}{_DEG}",
    re.IGNORECASE,
)
_BELOW_INCL_RE = re.compile(
    rf"(?:at most\s*{_NUM}{_DEG})"
    rf"|(?:{_NUM}{_DEG}\s*or (?:below|lower|less|fewer|colder|cooler))",
    re.IGNORECASE,
)
_BELOW_EXCL_RE = re.compile(
    rf"(?:below|under|less than|lower than|colder than|cooler than)\s*{_NUM}{_DEG}",
    re.IGNORECASE,
)
_EXACT_RE = re.compile(rf"\bbe\s*{_NUM}\s*°\s*F?\b", re.IGNORECASE)


def _first_group(match: re.Match) -> int:
    return int(next(g for g in match.groups() if g is not None))

_MONTHS = (
    "january february march april may june july august "
    "september october november december"
).split()
_DATE_RE = re.compile(
    rf"\b({'|'.join(_MONTHS)})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?",
    re.IGNORECASE,
)


class MarketDataError(RuntimeError):
    """Raised when Polymarket market data cannot be retrieved."""


class Bucket(tuple):
    """An inclusive integer-degree bucket, e.g. (90, 91) or (94, inf)."""

    __slots__ = ()

    def __new__(cls, low: float, high: float):
        return super().__new__(cls, (low, high))

    @property
    def low(self) -> float:
        return self[0]

    @property
    def high(self) -> float:
        return self[1]

    def continuous_bounds(self) -> tuple[float, float]:
        """Widen to the continuous interval that rounds into this bucket.

        Markets settle on the integer high, so bucket "90-91" is really
        [89.5, 91.5) in the continuous temperature the ensemble predicts.
        Skipping this step systematically understates the probability of
        narrow buckets, which is where most of the mispricing lives.
        """
        low = self.low - 0.5 if self.low != float("-inf") else float("-inf")
        high = self.high + 0.5 if self.high != float("inf") else float("inf")
        return low, high


def parse_bucket(question: str) -> Bucket | None:
    """Extract the temperature bucket from a market question."""
    text = question.replace("−", "-")

    m = _RANGE_RE.search(text)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
        return Bucket(min(low, high), max(low, high))

    m = _ABOVE_INCL_RE.search(text)
    if m:
        return Bucket(_first_group(m), float("inf"))

    m = _ABOVE_EXCL_RE.search(text)
    if m:
        return Bucket(_first_group(m) + 1, float("inf"))

    m = _BELOW_INCL_RE.search(text)
    if m:
        return Bucket(float("-inf"), _first_group(m))

    m = _BELOW_EXCL_RE.search(text)
    if m:
        return Bucket(float("-inf"), _first_group(m) - 1)

    m = _EXACT_RE.search(text)
    if m:
        value = int(m.group(1))
        return Bucket(value, value)

    return None


def parse_target_date(text: str, today: date | None = None) -> date | None:
    """Parse the settlement date named in a question or slug."""
    today = today or date.today()
    normalised = text.replace("-", " ")
    m = _DATE_RE.search(normalised)
    if not m:
        return None
    month = _MONTHS.index(m.group(1).lower()) + 1
    day = int(m.group(2))
    year = int(m.group(3)) if m.group(3) else today.year
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    # Undated questions near a year boundary refer to the nearer occurrence.
    if not m.group(3) and (parsed - today).days < -180:
        try:
            parsed = date(year + 1, month, day)
        except ValueError:
            return None
    return parsed


def _decode(value, default):
    """Gamma returns some list fields as JSON-encoded strings."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


@dataclass
class MarketQuote:
    market_id: str
    question: str
    slug: str
    city_key: str
    target_date: date
    bucket: Bucket
    yes_price: float
    volume: float
    liquidity: float
    end_date: str = ""

    @property
    def key(self) -> str:
        return f"{self.market_id}"


def build_quote(raw: dict, today: date | None = None) -> MarketQuote | None:
    """Turn one Gamma market record into a quote, or None if not applicable."""
    question = raw.get("question") or ""
    slug = raw.get("slug") or ""
    if not TEMPERATURE_RE.search(question):
        return None

    city: City | None = find_city(question) or find_city(slug)
    if city is None:
        return None

    bucket = parse_bucket(question)
    if bucket is None:
        return None

    target = parse_target_date(question, today) or parse_target_date(slug, today)
    if target is None:
        target = _date_from_end(raw.get("endDate"))
    if target is None:
        return None

    outcomes = [str(o).lower() for o in _decode(raw.get("outcomes"), [])]
    prices = [float(p) for p in _decode(raw.get("outcomePrices"), [])]
    if not prices:
        return None
    yes_index = outcomes.index("yes") if "yes" in outcomes else 0
    if yes_index >= len(prices):
        return None

    return MarketQuote(
        market_id=str(raw.get("id") or raw.get("conditionId") or slug),
        question=question,
        slug=slug,
        city_key=city.key,
        target_date=target,
        bucket=bucket,
        yes_price=prices[yes_index],
        volume=float(raw.get("volumeNum") or raw.get("volume") or 0.0),
        liquidity=float(raw.get("liquidityNum") or raw.get("liquidity") or 0.0),
        end_date=str(raw.get("endDate") or ""),
    )


def _date_from_end(end_date: str | None) -> date | None:
    if not end_date:
        return None
    try:
        parsed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).date()


def fetch_markets(session: requests.Session | None = None, max_pages: int = 6) -> list[dict]:
    """Page through open markets, most-traded first."""
    http = session or requests
    out: list[dict] = []
    for page in range(max_pages):
        params = {
            "closed": "false",
            "active": "true",
            "limit": PAGE_SIZE,
            "offset": page * PAGE_SIZE,
            "order": "volumeNum",
            "ascending": "false",
        }
        try:
            resp = http.get(GAMMA_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException as exc:
            raise MarketDataError(f"could not reach the Polymarket Gamma API: {exc}") from exc
        if not batch:
            break
        out.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
    return out


def temperature_quotes(raw_markets: list[dict], today: date | None = None) -> list[MarketQuote]:
    quotes = [build_quote(raw, today) for raw in raw_markets]
    return [q for q in quotes if q is not None]
