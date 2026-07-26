"""Learn each city's forecast bias from settled history.

This is the one piece the off-the-shelf bots skip, and it is the only part of
the pipeline that is genuinely proprietary to your own run: the public GFS run
is priced in the moment it is published, but *your* record of how that run has
missed at a specific station is not.

If the ensemble has averaged 3F hotter than the observed high at O'Hare, the
raw forecast is not the best estimate -- the forecast minus 3F is.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import ForecastRecord

#: Shrinkage constant. With `n` observations the measured bias is trusted
#: n/(n+K) of the way; at n=K it is halved. Keeps a three-day fluke from
#: shifting every probability in the book.
SHRINKAGE_K = 10.0


@dataclass
class CityBias:
    city_key: str
    samples: int
    mean_error: float  # positive => forecast ran hot
    mean_absolute_error: float
    applied_bias: float  # mean_error after shrinkage

    def as_row(self) -> str:
        return (
            f"{self.city_key:<10} n={self.samples:<4} "
            f"bias={self.mean_error:+.2f}F  mae={self.mean_absolute_error:.2f}F  "
            f"applied={self.applied_bias:+.2f}F"
        )


def learn_bias(
    records: list[ForecastRecord],
    min_samples: int = 5,
    shrinkage_k: float = SHRINKAGE_K,
) -> dict[str, CityBias]:
    """Compute a per-city bias from forecast records that have observations."""
    by_city: dict[str, list[float]] = {}
    for record in records:
        error = record.error
        if error is None:
            continue
        by_city.setdefault(record.city_key, []).append(error)

    out: dict[str, CityBias] = {}
    for city_key, errors in by_city.items():
        n = len(errors)
        if n < min_samples:
            continue
        mean_error = sum(errors) / n
        mae = sum(abs(e) for e in errors) / n
        applied = mean_error * (n / (n + shrinkage_k))
        out[city_key] = CityBias(
            city_key=city_key,
            samples=n,
            mean_error=round(mean_error, 3),
            mean_absolute_error=round(mae, 3),
            applied_bias=round(applied, 3),
        )
    return out


def bias_table(biases: dict[str, CityBias]) -> dict[str, float]:
    """Reduce learned biases to the plain dict the config carries."""
    return {key: value.applied_bias for key, value in sorted(biases.items())}
