"""Paper-trading ledger: positions, settlement and the equity curve.

Everything is simulated. `stake` is virtual money, `contracts` are virtual
contracts, and settlement is scored against observed temperature after the
fact. Nothing in this file can move funds.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timezone
from pathlib import Path

from .strategy import Signal

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent.parent / "simulation.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Position:
    id: str
    opened_at: str
    market_id: str
    question: str
    city_key: str
    target_date: str
    bucket_low: float
    bucket_high: float
    side: str
    price: float
    stake: float
    contracts: float
    model_prob: float
    edge: float
    forecast_mean: float
    forecast_sigma: float
    status: str = "open"
    actual_high: float | None = None
    actual_source: str | None = None
    payout: float | None = None
    pnl: float | None = None
    settled_at: str | None = None

    def wins_at(self, actual_high: float) -> bool:
        """Did this side pay out, given the observed high?

        Markets settle on the reported integer high, so the observation is
        rounded before it is compared with the bucket.
        """
        settled = round(actual_high)
        in_bucket = self.bucket_low <= settled <= self.bucket_high
        return in_bucket if self.side == "YES" else not in_bucket


@dataclass
class ForecastRecord:
    city_key: str
    target_date: str
    mean: float
    sigma: float
    members: int
    recorded_at: str
    actual_high: float | None = None
    actual_source: str | None = None

    @property
    def error(self) -> float | None:
        """Signed forecast error in F. Positive means the model ran hot."""
        if self.actual_high is None:
            return None
        return self.mean - self.actual_high


@dataclass
class Ledger:
    starting_bankroll: float = 1000.0
    cash: float = 1000.0
    positions: list[Position] = field(default_factory=list)
    forecasts: list[ForecastRecord] = field(default_factory=list)
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)

    # ---- persistence -------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None, starting_bankroll: float = 1000.0) -> "Ledger":
        path = Path(path) if path else DEFAULT_LEDGER_PATH
        if not path.exists():
            return cls(starting_bankroll=starting_bankroll, cash=starting_bankroll, path=path)
        data = json.loads(path.read_text())
        return cls(
            starting_bankroll=data.get("starting_bankroll", starting_bankroll),
            cash=data.get("cash", starting_bankroll),
            positions=[Position(**p) for p in data.get("positions", [])],
            forecasts=[ForecastRecord(**f) for f in data.get("forecasts", [])],
            path=path,
        )

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else (self.path or DEFAULT_LEDGER_PATH)
        payload = {
            "starting_bankroll": self.starting_bankroll,
            "cash": round(self.cash, 4),
            "updated_at": _now(),
            "positions": [asdict(p) for p in self.positions],
            "forecasts": [asdict(f) for f in self.forecasts],
        }
        target.write_text(json.dumps(payload, indent=2) + "\n")

    # ---- views -------------------------------------------------------

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "open"]

    def settled_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "settled"]

    def has_open_position(self, market_id: str) -> bool:
        return any(p.market_id == market_id and p.status == "open" for p in self.positions)

    @property
    def exposure(self) -> float:
        return sum(p.stake for p in self.open_positions())

    @property
    def equity(self) -> float:
        """Cash plus open positions marked at cost."""
        return self.cash + self.exposure

    @property
    def realised_pnl(self) -> float:
        return sum(p.pnl or 0.0 for p in self.settled_positions())

    def stats(self) -> dict:
        settled = self.settled_positions()
        wins = [p for p in settled if (p.pnl or 0) > 0]
        staked = sum(p.stake for p in settled)
        return {
            "starting_bankroll": round(self.starting_bankroll, 2),
            "cash": round(self.cash, 2),
            "exposure": round(self.exposure, 2),
            "equity": round(self.equity, 2),
            "realised_pnl": round(self.realised_pnl, 2),
            "return_pct": round(100 * (self.equity / self.starting_bankroll - 1), 2)
            if self.starting_bankroll
            else 0.0,
            "open_positions": len(self.open_positions()),
            "settled_positions": len(settled),
            "win_rate": round(len(wins) / len(settled), 4) if settled else None,
            "roi_on_settled": round(self.realised_pnl / staked, 4) if staked else None,
        }

    # ---- mutations ---------------------------------------------------

    def record_forecast(self, city_key: str, target_date: date, mean: float, sigma: float, members: int) -> None:
        stamp = target_date.isoformat()
        for existing in self.forecasts:
            if existing.city_key == city_key and existing.target_date == stamp:
                existing.mean = mean
                existing.sigma = sigma
                existing.members = members
                existing.recorded_at = _now()
                return
        self.forecasts.append(
            ForecastRecord(
                city_key=city_key,
                target_date=stamp,
                mean=mean,
                sigma=sigma,
                members=members,
                recorded_at=_now(),
            )
        )

    def open_position(self, signal: Signal) -> Position | None:
        """Book a paper fill at the current quote. Returns None if unaffordable."""
        if signal.stake > self.cash:
            return None
        position = Position(
            id=uuid.uuid4().hex[:12],
            opened_at=_now(),
            market_id=signal.quote.market_id,
            question=signal.quote.question,
            city_key=signal.quote.city_key,
            target_date=signal.quote.target_date.isoformat(),
            bucket_low=signal.quote.bucket.low,
            bucket_high=signal.quote.bucket.high,
            side=signal.side,
            price=round(signal.price, 4),
            stake=round(signal.stake, 2),
            contracts=round(signal.contracts, 4),
            model_prob=round(signal.model_prob, 4),
            edge=round(signal.edge, 4),
            forecast_mean=round(signal.forecast_mean, 2),
            forecast_sigma=round(signal.forecast_sigma, 2),
        )
        self.cash = round(self.cash - position.stake, 4)
        self.positions.append(position)
        return position

    def settle(self, city_key: str, target_date: date, actual_high: float,
               source: str = "station") -> list[Position]:
        """Settle every open position on a city/date and bank the payouts."""
        stamp = target_date.isoformat()
        settled: list[Position] = []
        for position in self.open_positions():
            if position.city_key != city_key or position.target_date != stamp:
                continue
            won = position.wins_at(actual_high)
            payout = round(position.contracts, 4) if won else 0.0
            position.status = "settled"
            position.actual_high = actual_high
            position.actual_source = source
            position.payout = payout
            position.pnl = round(payout - position.stake, 4)
            position.settled_at = _now()
            self.cash = round(self.cash + payout, 4)
            settled.append(position)

        for record in self.forecasts:
            if record.city_key == city_key and record.target_date == stamp:
                record.actual_high = actual_high
                record.actual_source = source
        return settled

    def observed_counts(self, source: str = "station") -> dict[str, int]:
        """How many forecasts per city have been scored against a real
        observation. This is the counter the trading guard reads: a city with
        no independent history has no measured bias, so there is nothing to
        correct and no business betting on it."""
        counts: dict[str, int] = {}
        for record in self.forecasts:
            if record.actual_high is not None and record.actual_source == source:
                counts[record.city_key] = counts.get(record.city_key, 0) + 1
        return counts

    def pending_settlements(self, today: date | None = None) -> list[tuple[str, date]]:
        """City/date pairs that are in the past and still unsettled."""
        today = today or date.today()
        pending: set[tuple[str, str]] = set()
        for position in self.open_positions():
            if date.fromisoformat(position.target_date) < today:
                pending.add((position.city_key, position.target_date))
        for record in self.forecasts:
            if record.actual_high is None and date.fromisoformat(record.target_date) < today:
                pending.add((record.city_key, record.target_date))
        return sorted((c, date.fromisoformat(d)) for c, d in pending)
