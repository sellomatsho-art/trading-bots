import json

import pytest

from weatherbot.config import DEFAULT_CONFIG_PATH, Config


def test_defaults_are_conservative():
    cfg = Config()
    assert cfg.kelly_fraction <= 0.5, "never full Kelly on a self-estimated probability"
    assert cfg.max_position_fraction <= 0.1
    assert 0 < cfg.min_price < cfg.max_price < 1


def test_load_missing_file_falls_back_to_defaults(tmp_path):
    cfg = Config.load(tmp_path / "nope.json")
    assert cfg.bankroll == Config().bankroll


def test_unknown_keys_are_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"bankrol": 50}))
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(path)


def test_save_writes_back_to_the_file_it_was_loaded_from(tmp_path):
    """Regression: save() used to default to the packaged config.json and
    overwrite the real project settings during an unrelated run."""
    path = tmp_path / "config.json"
    Config(bankroll=250.0).save(path)

    cfg = Config.load(path)
    cfg.bias_correction["nyc"] = 2.5
    cfg.save()

    assert json.loads(path.read_text())["bias_correction"] == {"nyc": 2.5}
    assert cfg.path == path


def test_save_does_not_touch_the_default_config(tmp_path):
    before = DEFAULT_CONFIG_PATH.read_text() if DEFAULT_CONFIG_PATH.exists() else None
    cfg = Config.load(tmp_path / "config.json")
    cfg.bankroll = 12345.0
    cfg.save()
    after = DEFAULT_CONFIG_PATH.read_text() if DEFAULT_CONFIG_PATH.exists() else None
    assert before == after


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "config.json"
    original = Config(cities=["nyc"], min_edge=0.11, bias_correction={"nyc": -1.25})
    original.save(path)
    assert Config.load(path) == original
