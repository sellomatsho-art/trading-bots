"""`weatherbot reset` — archives the paper ledger and starts a clean one.

Exists because the first live run built 20 settled positions and a +28%
return on a forecast sitting ~5F above the market, settled against the same
grid the forecast came from. That equity is not a measurement of anything and
must not be carried into a calibrated run.
"""

import json
from datetime import date

import pytest

from weatherbot.cli import main
from weatherbot.config import Config
from weatherbot.ledger import Ledger


def _seeded(tmp_path):
    cfg_path = tmp_path / "config.json"
    Config(bankroll=1000.0).save(cfg_path)
    led_path = tmp_path / "sim.json"
    ledger = Ledger(starting_bankroll=1000, cash=940, path=led_path)
    ledger.record_forecast("nyc", date(2026, 8, 1), 87.0, 2.0, 31)
    ledger.settle("nyc", date(2026, 8, 1), actual_high=82.0, source="grid")
    ledger.save()
    return cfg_path, led_path


def _run(cfg_path, led_path, *extra):
    return main(["--config", str(cfg_path), "--ledger", str(led_path), "reset", *extra])


def test_reset_archives_rather_than_deletes(tmp_path):
    """The old ledger is the evidence for whatever made you reset it."""
    cfg_path, led_path = _seeded(tmp_path)
    before = led_path.read_text()

    assert _run(cfg_path, led_path, "--yes") == 0

    archives = [p for p in tmp_path.glob("sim.*.json")]
    assert len(archives) == 1, [p.name for p in archives]
    assert archives[0].read_text() == before


def test_reset_leaves_a_fresh_ledger_at_the_configured_bankroll(tmp_path):
    cfg_path, led_path = _seeded(tmp_path)
    _run(cfg_path, led_path, "--yes")

    fresh = Ledger.load(led_path)
    assert fresh.positions == [] and fresh.forecasts == []
    assert fresh.cash == 1000.0 and fresh.starting_bankroll == 1000.0
    assert fresh.stats()["equity"] == 1000.0


def test_reset_refuses_non_interactively_without_yes(tmp_path, monkeypatch, capsys):
    """A destructive default must not fire from a cron job or a pipe."""
    cfg_path, led_path = _seeded(tmp_path)
    before = led_path.read_text()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert _run(cfg_path, led_path) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert led_path.read_text() == before, "ledger untouched"
    assert list(tmp_path.glob("sim.*.json")) == [], "nothing archived either"
    assert json.loads(before)["forecasts"], "fixture really did have history"


def test_reset_on_a_missing_ledger_is_a_no_op(tmp_path, capsys):
    cfg_path = tmp_path / "config.json"
    Config(bankroll=1000.0).save(cfg_path)
    missing = tmp_path / "nope.json"

    assert main(["--config", str(cfg_path), "--ledger", str(missing), "reset", "--yes"]) == 0
    assert "nothing to reset" in capsys.readouterr().out
    assert not missing.exists(), "no ledger is conjured out of nothing"


def test_reset_reports_how_much_of_the_history_was_station_scored(tmp_path, capsys):
    """The number that matters when deciding whether to reset: grid-settled
    history cannot calibrate anything."""
    cfg_path, led_path = _seeded(tmp_path)
    _run(cfg_path, led_path, "--yes")

    out = capsys.readouterr().out
    assert "1 forecast records, 0 scored against a station" in out


def test_a_declined_confirmation_changes_nothing(tmp_path, monkeypatch):
    cfg_path, led_path = _seeded(tmp_path)
    before = led_path.read_text()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "n")

    assert _run(cfg_path, led_path) == 0
    assert led_path.read_text() == before
    assert list(tmp_path.glob("sim.*.json")) == []


def test_confirming_at_the_prompt_performs_the_reset(tmp_path, monkeypatch):
    cfg_path, led_path = _seeded(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    assert _run(cfg_path, led_path) == 0
    assert Ledger.load(led_path).positions == []
    assert len(list(tmp_path.glob("sim.*.json"))) == 1
