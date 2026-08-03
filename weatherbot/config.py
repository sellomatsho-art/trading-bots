"""Configuration loading.

Everything here is trading *parameters* only. There is deliberately no field
for a wallet key, seed phrase, or API secret of any kind -- see README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class Config:
    #: Cities to scan, by key in weatherbot.cities.CITIES.
    cities: list[str] = field(default_factory=lambda: ["nyc", "chicago", "la", "miami", "austin"])

    #: Starting virtual bankroll in USD.
    bankroll: float = 1000.0

    #: Minimum |model probability - market price| to consider a trade.
    min_edge: float = 0.08

    #: Ignore markets priced outside this band; the tails are where thin books
    #: and settlement quirks do the most damage.
    min_price: float = 0.05
    max_price: float = 0.95

    #: Fraction of full Kelly to actually stake. Full Kelly on a probability
    #: you estimated yourself is a good way to go broke.
    kelly_fraction: float = 0.25

    #: Hard cap on any single position as a share of bankroll.
    max_position_fraction: float = 0.05

    #: Max concurrent open paper positions.
    max_open_positions: int = 20

    #: Skip markets with less than this much reported 24h volume (USD).
    min_volume: float = 500.0

    #: How far ahead to look for markets, in days.
    forecast_horizon_days: int = 3

    #: Probability model: "empirical", "gaussian", or "blend".
    probability_model: str = "blend"

    #: Weight on the raw ensemble histogram in "blend" mode.
    empirical_weight: float = 0.5

    #: Floor on the ensemble spread, in F. With 31 members a tight cluster
    #: produces overconfident buckets; this keeps probabilities honest.
    min_sigma_f: float = 1.5

    #: Refuse to trade a city until its forecast has been scored against
    #: enough independent station observations to measure a bias. Without
    #: this the bot bets hardest exactly when it knows least -- on 3 Aug 2026
    #: it staked the per-position cap on six signals while sitting ~5F above
    #: the market at both cities it was trading.
    require_calibration: bool = True

    #: Independent observations a city needs before it may be traded.
    min_calibration_samples: int = 5

    #: Per-city forecast bias in F, learned by `weatherbot calibrate` and
    #: subtracted from the raw ensemble. Positive => model runs hot.
    bias_correction: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Where this config came from, so `save()` writes back to the same
        # file instead of silently clobbering the default one.
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            cfg = cls()
            cfg._path = path
            return cfg
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg = cls(**data)
        cfg._path = path
        return cfg

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else (self._path or DEFAULT_CONFIG_PATH)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n")
        self._path = target
