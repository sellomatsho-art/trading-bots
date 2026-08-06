from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from weatherbot.cities import CITIES
from weatherbot.markets import Bucket
from weatherbot.scalp import (
    DayState,
    bucket_probability,
    day_state,
    late_rise_samples,
    rise_distribution,
)

LA = CITIES["la"]
TZ = ZoneInfo(LA.timezone)
TODAY = date(2026, 8, 6)
CUTOFF = 18


def _series(day, temps_by_hour):
    return [(datetime(day.year, day.month, day.day, h, 0, tzinfo=TZ), t)
            for h, t in sorted(temps_by_hour.items())]


def _typical_day(day, peak=80.0, evening=None):
    """A normal diurnal curve: cool morning, mid-afternoon peak, cooling off."""
    curve = {6: peak - 12, 9: peak - 8, 12: peak - 3, 14: peak, 16: peak - 1,
             19: peak - 5 if evening is None else evening, 22: peak - 9}
    return _series(day, curve)


# --- today's state ---------------------------------------------------------

def test_day_state_takes_the_max_before_the_cutoff():
    state = day_state(_typical_day(TODAY, peak=84.0), LA, TODAY, CUTOFF)
    assert state.max_so_far == pytest.approx(84.0)
    assert state.rounded_max == 84
    assert state.last_reading_at.hour == 16, "16:00 is the last reading before 18"


def test_day_state_ignores_readings_at_or_after_the_cutoff():
    """A 19:00 spike must not count toward the max we are treating as known."""
    state = day_state(_typical_day(TODAY, peak=80.0, evening=95.0), LA, TODAY, CUTOFF)
    assert state.max_so_far == pytest.approx(80.0)


def test_day_state_is_none_before_any_reading():
    assert day_state([], LA, TODAY, CUTOFF) is None


def test_day_state_is_none_when_the_cutoff_has_not_been_reached():
    """Pre-cutoff the peak may still be ahead, so there is nothing to scalp."""
    morning = _series(TODAY, {6: 60.0, 9: 66.0})
    assert day_state(morning, LA, TODAY, cutoff_hour=5) is None


def test_day_state_ignores_other_days():
    series = _typical_day(TODAY - timedelta(days=1), peak=99.0) + _typical_day(TODAY, peak=70.0)
    assert day_state(series, LA, TODAY, CUTOFF).max_so_far == pytest.approx(70.0)


# --- the measured residual risk -------------------------------------------

def test_late_rise_is_zero_on_a_normal_cooling_evening():
    series = []
    for i in range(1, 11):
        series += _typical_day(TODAY - timedelta(days=i), peak=80.0)
    assert late_rise_samples(series, CUTOFF, TODAY) == [0] * 10


def test_late_rise_catches_an_evening_warm_up():
    """A Santa Ana / chinook evening: the high arrives after the cutoff.

    This is the risk the strategy actually carries, and the reason nothing
    here is allowed to assume the day is over at 6pm.
    """
    series = _typical_day(TODAY - timedelta(days=1), peak=80.0, evening=83.4)
    assert late_rise_samples(series, CUTOFF, TODAY) == [3]


def test_late_rise_excludes_today():
    """Today's day is not over, so its max is not a final high."""
    series = _typical_day(TODAY, peak=80.0, evening=90.0)
    assert late_rise_samples(series, CUTOFF, TODAY) == []


def test_late_rise_skips_days_with_no_pre_cutoff_coverage():
    series = _series(TODAY - timedelta(days=1), {20: 75.0, 22: 71.0})
    assert late_rise_samples(series, CUTOFF, TODAY) == []


def test_late_rise_samples_are_never_negative():
    """The max is monotone in time; a negative sample would mean a bug."""
    series = []
    for i in range(1, 15):
        series += _typical_day(TODAY - timedelta(days=i), peak=70.0 + i)
    assert all(s >= 0 for s in late_rise_samples(series, CUTOFF, TODAY))


# --- the distribution ------------------------------------------------------

def test_a_clean_history_still_leaves_room_for_a_rise():
    """30 quiet evenings is evidence, not proof. P(rise) must stay > 0."""
    dist = rise_distribution([0] * 30)
    assert dist.probability(0) > 0.9
    assert dist.probability(1) > 0.0, "smoothing keeps the tail reachable"
    assert dist.at_least(1) < 0.1


def test_the_distribution_reflects_a_history_of_rises():
    dist = rise_distribution([0] * 6 + [1] * 3 + [2])
    assert dist.probability(0) > dist.probability(1) > dist.probability(2)
    assert dist.at_least(1) > 0.3


def test_an_empty_history_yields_no_probability_mass():
    """No measurement means no opinion -- the caller must gate on samples."""
    dist = rise_distribution([])
    assert dist.samples == 0
    assert dist.probability(0) == 0.0


# --- pricing ---------------------------------------------------------------

def _state(max_so_far):
    return DayState(city_key="la", target_date=TODAY, max_so_far=max_so_far,
                    last_reading_at=datetime(2026, 8, 6, 17, tzinfo=TZ), readings=12)


def test_buckets_below_the_observed_max_are_already_impossible():
    """The core of the edge, and the part that needs no model: the high
    cannot come in below a temperature the station has already reported."""
    dist = rise_distribution([0] * 20)
    assert bucket_probability(_state(84.0), Bucket(80, 81), dist) == pytest.approx(0.005)
    assert bucket_probability(_state(84.0), Bucket(float("-inf"), 83), dist) == pytest.approx(0.005)


def test_the_bucket_holding_the_observed_max_carries_most_of_the_mass():
    dist = rise_distribution([0] * 20)
    p = bucket_probability(_state(84.0), Bucket(84, 85), dist)
    assert p > 0.9


def test_a_history_of_evening_rises_moves_mass_upward():
    quiet = rise_distribution([0] * 20)
    gusty = rise_distribution([0] * 10 + [1] * 6 + [2] * 4)
    here = Bucket(84, 84)
    above = Bucket(85, float("inf"))
    assert bucket_probability(_state(84.0), here, gusty) < bucket_probability(_state(84.0), here, quiet)
    assert bucket_probability(_state(84.0), above, gusty) > bucket_probability(_state(84.0), above, quiet)


def test_probabilities_are_clamped_short_of_certainty():
    """Kelly divides by (1 - price); a probability of exactly 1 explodes it."""
    dist = rise_distribution([0] * 500
                             )
    p = bucket_probability(_state(84.0), Bucket(84, 84), dist)
    assert p <= 0.995


def test_the_observed_max_rounds_the_way_settlement_does():
    assert _state(83.6).rounded_max == 84
    assert _state(83.4).rounded_max == 83
    dist = rise_distribution([0] * 20)
    assert bucket_probability(_state(83.6), Bucket(84, 85), dist) > 0.9


def test_mass_sums_to_about_one_across_a_covering_ladder():
    dist = rise_distribution([0] * 12 + [1] * 5 + [2] * 3)
    state = _state(84.0)
    buckets = [Bucket(float("-inf"), 83), Bucket(84, 84), Bucket(85, 85),
               Bucket(86, 86), Bucket(87, float("inf"))]
    total = sum(bucket_probability(state, b, dist) for b in buckets)
    assert 0.98 < total < 1.03, total
