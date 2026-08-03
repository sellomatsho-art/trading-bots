import pytest

from weatherbot.calibration import bias_table, learn_bias
from weatherbot.ledger import ForecastRecord


def record(city, mean, actual, source="station"):
    return ForecastRecord(
        city_key=city,
        target_date="2026-07-26",
        mean=mean,
        sigma=1.0,
        members=31,
        recorded_at="",
        actual_high=actual,
        actual_source=source,
    )


def test_learn_bias_detects_a_consistently_hot_forecast():
    """The worked example: NWS calls 72F in Chicago, it verifies near 69F."""
    records = [record("chicago", 72.0, 69.0) for _ in range(20)]
    biases = learn_bias(records)
    assert biases["chicago"].mean_error == pytest.approx(3.0)
    assert biases["chicago"].mean_absolute_error == pytest.approx(3.0)
    assert biases["chicago"].samples == 20


def test_shrinkage_pulls_a_thin_sample_toward_zero():
    few = learn_bias([record("nyc", 72.0, 69.0) for _ in range(5)])["nyc"]
    many = learn_bias([record("nyc", 72.0, 69.0) for _ in range(200)])["nyc"]
    assert few.applied_bias < many.applied_bias < 3.0
    assert few.applied_bias == pytest.approx(3.0 * 5 / 15)
    assert many.applied_bias == pytest.approx(3.0, abs=0.15)


def test_cities_below_the_sample_floor_are_not_reported():
    records = [record("miami", 90.0, 88.0) for _ in range(3)]
    assert learn_bias(records, min_samples=5) == {}
    assert "miami" in learn_bias(records, min_samples=3)


def test_records_without_an_observation_are_ignored():
    records = [record("nyc", 72.0, 69.0) for _ in range(6)]
    records += [
        ForecastRecord("nyc", "2026-07-27", 99.0, 1.0, 31, "", actual_high=None)
        for _ in range(50)
    ]
    assert learn_bias(records)["nyc"].samples == 6


def test_a_cold_bias_comes_back_negative():
    biases = learn_bias([record("denver", 65.0, 68.0) for _ in range(20)])
    assert biases["denver"].mean_error == pytest.approx(-3.0)
    assert biases["denver"].applied_bias < 0


def test_errors_that_cancel_leave_no_bias_but_a_real_mae():
    records = [record("la", 75.0, 70.0), record("la", 65.0, 70.0)] * 5
    bias = learn_bias(records)["la"]
    assert bias.mean_error == pytest.approx(0.0)
    assert bias.mean_absolute_error == pytest.approx(5.0)


def test_bias_table_reduces_to_a_plain_config_dict():
    records = [record("nyc", 72.0, 69.0) for _ in range(20)]
    records += [record("miami", 90.0, 91.0) for _ in range(20)]
    table = bias_table(learn_bias(records))
    assert set(table) == {"miami", "nyc"}
    assert all(isinstance(v, float) for v in table.values())


def test_grid_sourced_observations_cannot_move_a_bias():
    """The 3 Aug 2026 finding, pinned.

    Settling against the same Open-Meteo grid the forecast is drawn from
    means a systematic grid error appears on both sides of the subtraction
    and largely cancels. Those records must not feed calibration at all,
    however many of them there are.
    """
    grid = [record("nyc", 87.0, 82.0, source="grid") for _ in range(200)]
    assert learn_bias(grid) == {}


def test_a_station_record_still_learns_alongside_grid_noise():
    records = [record("nyc", 87.0, 82.0, source="grid") for _ in range(200)]
    records += [record("nyc", 87.0, 82.0, source="station") for _ in range(10)]
    bias = learn_bias(records)["nyc"]
    assert bias.samples == 10, "only the station rows count"
    assert bias.mean_error == pytest.approx(5.0)


def test_records_with_no_source_are_not_trusted():
    """Rows written before provenance existed have an unknown source."""
    legacy = [ForecastRecord("nyc", f"2026-07-{i:02d}", 87.0, 1.0, 31, "",
                             actual_high=82.0) for i in range(1, 21)]
    assert learn_bias(legacy) == {}
