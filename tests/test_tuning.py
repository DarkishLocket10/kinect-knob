"""Live-tuning schema and manager: every dashboard slider goes through here."""
import json

import pytest

from kinectknob.config import AppConfig
from kinectknob.tuning import PAIRED, TUNABLES, Tuning


def make_tuning(tmp_path, cfg=None):
    return Tuning(cfg or AppConfig(), path=str(tmp_path / "tuning.json"))


def test_schema_keys_resolve_and_defaults_in_range():
    cfg = AppConfig()
    for t in TUNABLES:
        section, _, attr = t.key.partition(".")
        value = getattr(getattr(cfg, section), attr)  # raises if the key is stale
        if t.kind == "bool":
            assert isinstance(value, bool), t.key
        else:
            assert t.min <= value <= t.max, f"{t.key} default {value} outside slider range"
            assert isinstance(value, (int, float)) and not isinstance(value, bool), t.key


def test_paired_keys_exist_in_schema():
    keys = {t.key for t in TUNABLES}
    for lower, upper, gap in PAIRED:
        assert lower in keys and upper in keys
        assert gap > 0


def test_set_applies_live_and_clamps(tmp_path):
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    assert tn.set_value("knob.deadband_deg", 8.0) == 8.0
    assert cfg.knob.deadband_deg == 8.0            # live on the shared config
    assert tn.set_value("knob.deadband_deg", 999) == 15.0   # clamped to max
    assert tn.set_value("knob.engage_frames", 7.6) == 8     # int coercion
    assert tn.set_value("swipe.enabled", 0) is False        # bool coercion


def test_pair_rails_keep_thresholds_apart(tmp_path):
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    # Push engage above release: railed to release - gap.
    applied = tn.set_value("knob.engage_pinch", 0.60)
    assert applied <= cfg.knob.release_pinch - 0.05 + 1e-9
    # Push depth-max below depth-min: railed to min + gap.
    tn.set_value("gate.depth_min_m", 1.5)
    applied = tn.set_value("gate.depth_max_m", 1.0)
    assert applied >= 1.5 + 0.2 - 1e-9


def test_persists_deltas_and_reloads(tmp_path):
    cfg1 = AppConfig()
    tn1 = make_tuning(tmp_path, cfg1)
    tn1.set_value("knob.engage_pinch", 0.30)
    saved = json.loads((tmp_path / "tuning.json").read_text())
    assert saved == {"knob.engage_pinch": 0.30}     # deltas only, not everything

    cfg2 = AppConfig()
    make_tuning(tmp_path, cfg2)                     # fresh process, same file
    assert cfg2.knob.engage_pinch == 0.30


def test_reset_restores_baseline_and_removes_file(tmp_path):
    cfg = AppConfig()
    base = cfg.knob.engage_pinch
    tn = make_tuning(tmp_path, cfg)
    tn.set_value("knob.engage_pinch", 0.30)
    tn.reset()
    assert cfg.knob.engage_pinch == base
    assert not (tmp_path / "tuning.json").exists()


def test_unknown_key_rejected(tmp_path):
    tn = make_tuning(tmp_path)
    with pytest.raises(KeyError):
        tn.set_value("knob.does_not_exist", 1)


def test_stale_file_keys_ignored(tmp_path):
    (tmp_path / "tuning.json").write_text(
        json.dumps({"removed.setting": 5, "knob.deadband_deg": 6.0})
    )
    cfg = AppConfig()
    make_tuning(tmp_path, cfg)
    assert cfg.knob.deadband_deg == 6.0             # good key applied, stale ignored


def test_schema_exposes_value_and_default(tmp_path):
    tn = make_tuning(tmp_path)
    tn.set_value("gate.min_score", 0.7)
    row = next(r for r in tn.schema() if r["key"] == "gate.min_score")
    assert row["value"] == 0.7 and row["default"] == 0.55
    assert row["help"] and row["label"] and row["group"]
