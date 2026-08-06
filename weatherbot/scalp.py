"""Post-event scalp: price the day's high once it is mostly already known.

The band-fade and forecast strategies both failed the same test — neither had
an answer to "who is on the other side, and why are they willing to lose?"
This one does.

A city's daily high is set at the diurnal peak, typically early-to-mid
afternoon. The market does not resolve until the local day ends. In the hours
between, the outcome is close to determined while the market may still be
priced as though it were not. The reason that persists is not a mispriced
model — it is that a thin market at 7pm has nobody watching it.

Two things make this different in kind from what came before:

1. **It is an observation, not a forecast.** The max so far is a fact from the
   station feed. We are not claiming to out-predict anyone.
2. **The residual risk is measured, not assumed.** The only thing that can
   still move the outcome is a further rise after the cutoff. That quantity is
   directly measurable from the same station's history — pull the last N days,
   compare the max as of the cutoff with the max over the whole day, and the
   distribution of the difference IS the risk. No prior required, and no
   waiting weeks to accumulate it: NWS serves arbitrary past ranges, so the
   estimate is available on the first run.

What could still go wrong, stated up front rather than discovered later:

- **Late rises are real.** Downslope wind (Denver chinook, LA Santa Ana) can
  push the temperature up in the evening. That is exactly what the measured
  distribution is for, and why nothing here assumes P=1.
- **Settlement source.** These probabilities are for the NWS station max. If
  the market resolves on the 6-hourly climate-report max instead, the two can
  differ by a degree, which on a 2-degree bucket is the whole edge.
- **Quoted price is not executable price.** A stale 0.70 on an untraded book
  is not a fill. Depth is not modelled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from .cities import City
from .forecast import ForecastError, fetch_station_series
from .markets import Bucket

#: Never let a measured probability reach certainty. Even a clean history of
#: "the high never rose after 6pm" is a finite sample, and a bucket priced at
#: 1.0 makes Kelly divide by zero.
PROB_FLOOR = 0.005
PROB_CEIL = 0.995


@dataclass
class DayState:
    """What the station has reported so far today."""

    city_key: str
    target_date: date
    max_so_far: float
    last_reading_at: datetime  # local
    readings: int

    @property
    def rounded_max(self) -> int:
        """Markets settle on the integer high, so reason in that space."""
        return round(self.max_so_far)


def city_now(city: City) -> datetime:
    """Wall-clock time in the city, which is the only clock that matters.

    Everything about this strategy is anchored to the city's own day: when
    its peak has passed, which date its markets settle on, which readings
    count. The machine's clock is irrelevant and actively misleading — run
    from UTC+2 at 00:42, `date.today()` returns tomorrow while every US city
    is still in yesterday afternoon.
    """
    return datetime.now(ZoneInfo(city.timezone))


def day_state(series: list[tuple[datetime, float]], city: City,
              target_date: date, cutoff_hour: int,
              now_local: datetime) -> DayState | None:
    """Reduce today's readings up to the cutoff into a state, or None.

    None means the day is not yet scalpable.

    `now_local` is required rather than derived, and the clock check is the
    reason. An earlier version only asked whether any readings existed
    before the cutoff hour — which is true at 10am, so it would happily
    report a "max so far" at mid-morning and price markets as though the
    day were settled. The peak had not happened yet. The docstring claimed
    the clock was checked; the code never looked at it.
    """
    if now_local.date() != target_date or now_local.hour < cutoff_hour:
        return None
    today = [(when, t) for when, t in series
             if when.date() == target_date and when.hour < cutoff_hour]
    if not today:
        return None
    last, _ = today[-1]
    return DayState(
        city_key=city.key,
        target_date=target_date,
        max_so_far=max(t for _, t in today),
        last_reading_at=last,
        readings=len(today),
    )


def late_rise_samples(series: list[tuple[datetime, float]], cutoff_hour: int,
                      today: date) -> list[int]:
    """Per past day: how many whole degrees the high rose after the cutoff.

    Measured in rounded-degree space because that is what settles. The max is
    monotone in time, so every sample is >= 0.

    Today is excluded: its day is not over, so its "final" high is not final.
    """
    by_day: dict[date, list[tuple[datetime, float]]] = {}
    for when, temp in series:
        if when.date() < today:
            by_day.setdefault(when.date(), []).append((when, temp))

    samples: list[int] = []
    for day, readings in sorted(by_day.items()):
        before = [t for when, t in readings if when.hour < cutoff_hour]
        if not before:
            continue  # no pre-cutoff coverage; the pair would be meaningless
        after_all = [t for _, t in readings]
        samples.append(round(max(after_all)) - round(max(before)))
    return samples


@dataclass
class RiseDistribution:
    """P(the high rises k more degrees after the cutoff), k = 0, 1, 2, ...

    Laplace-smoothed: a run of days where nothing rose is evidence, not proof,
    and an unsmoothed zero would price the tail buckets at exactly 0.
    """

    counts: dict[int, int]
    samples: int
    max_rise: int

    def probability(self, rise: int) -> float:
        if self.samples <= 0:
            return 0.0
        # One pseudo-count spread over the observed support plus one slot
        # beyond it, so "never seen" is small rather than impossible.
        support = self.max_rise + 2
        return (self.counts.get(rise, 0) + 1.0) / (self.samples + support)

    def at_least(self, rise: int) -> float:
        if rise <= 0:
            return 1.0
        return sum(self.probability(k) for k in range(rise, self.max_rise + 2))


def rise_distribution(samples: list[int]) -> RiseDistribution:
    counts: dict[int, int] = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    return RiseDistribution(
        counts=counts,
        samples=len(samples),
        max_rise=max(samples) if samples else 0,
    )


def bucket_probability(state: DayState, bucket: Bucket,
                       dist: RiseDistribution) -> float:
    """P(the settled integer high lands in this bucket), given today's max.

    The final high is `rounded_max + R` where R is the measured rise. Buckets
    entirely below the max so far are already impossible — that is the whole
    source of the edge, and the one thing here that needs no model at all.
    """
    base = state.rounded_max
    total = 0.0
    for rise in range(0, dist.max_rise + 2):
        final = base + rise
        if bucket.low <= final <= bucket.high:
            total += dist.probability(rise)
    return min(max(total, PROB_FLOOR), PROB_CEIL)


def fetch_scalp_inputs(city: City, target_date: date, cutoff_hour: int,
                       history_days: int,
                       session: requests.Session | None = None,
                       now_local: datetime | None = None,
                       ) -> tuple[DayState | None, RiseDistribution, list]:
    """Today's state, the measured rise risk, and any days that came back short.

    Truncated days are surfaced rather than swallowed: a short history still
    produces a confident-looking distribution, which is the worst kind of
    wrong here.
    """
    if not city.station_id:
        raise ForecastError(f"no independent station for {city.key}")
    start = target_date - timedelta(days=history_days)
    series, truncated = fetch_station_series(city, start, target_date, session=session)
    state = day_state(series, city, target_date, cutoff_hour,
                      now_local or city_now(city))
    dist = rise_distribution(late_rise_samples(series, cutoff_hour, target_date))
    return state, dist, truncated
