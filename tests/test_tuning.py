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


# -- 2026-10-30: hand-tuned values must survive anything ----------------------

def test_a_value_set_to_the_baseline_is_still_saved(tmp_path):
    """A tuned value that happens to equal today's default must not drift
    when a code default or env var changes later."""
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    tn.set_value("gate.min_score", cfg.gate.min_score)
    saved = json.loads((tmp_path / "tuning.json").read_text())
    assert saved == {"gate.min_score": cfg.gate.min_score}
    cfg2 = AppConfig()
    cfg2.gate.min_score = 0.9                    # "the default changed"
    make_tuning(tmp_path, cfg2)
    assert cfg2.gate.min_score == cfg.gate.min_score


def test_unknown_keys_survive_a_save(tmp_path):
    (tmp_path / "tuning.json").write_text(json.dumps({"fist.hold_s": 0.8}))
    tn = make_tuning(tmp_path)
    tn.set_value("knob.deadband_deg", 6.0)
    saved = json.loads((tmp_path / "tuning.json").read_text())
    assert saved == {"fist.hold_s": 0.8, "knob.deadband_deg": 6.0}


def test_an_out_of_range_saved_value_is_kept_in_the_file(tmp_path):
    (tmp_path / "tuning.json").write_text(json.dumps({"knob.deadband_deg": 999}))
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    assert cfg.knob.deadband_deg == 15.0         # applied clamped
    tn.set_value("gate.min_score", 0.7)          # an unrelated save
    saved = json.loads((tmp_path / "tuning.json").read_text())
    assert saved["knob.deadband_deg"] == 999     # original kept
    tn.set_value("knob.deadband_deg", 5.0)       # until the user moves it
    assert json.loads((tmp_path / "tuning.json").read_text())["knob.deadband_deg"] == 5.0


def test_reset_backs_up_and_restore_undoes_it(tmp_path):
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    tn.set_value("knob.engage_pinch", 0.30)
    tn.set_value("swipe.invert", True)
    backup = tn.reset()
    assert backup and (tmp_path / backup).is_file()
    assert cfg.knob.engage_pinch != 0.30
    tn.restore()
    assert cfg.knob.engage_pinch == 0.30 and cfg.swipe.invert is True
    assert json.loads((tmp_path / "tuning.json").read_text()) == {
        "knob.engage_pinch": 0.30, "swipe.invert": True}


def test_restore_only_reads_its_own_backups(tmp_path):
    tn = make_tuning(tmp_path)
    with pytest.raises(FileNotFoundError):
        tn.restore("../../etc/passwd")


def test_reset_key_unpins_one_setting(tmp_path):
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    tn.set_value("knob.deadband_deg", 6.0)
    tn.set_value("gate.min_score", 0.7)
    tn.reset_key("knob.deadband_deg")
    assert json.loads((tmp_path / "tuning.json").read_text()) == {"gate.min_score": 0.7}


def test_export_import_round_trip(tmp_path):
    tn = Tuning(AppConfig(), path=str(tmp_path / "a.json"))
    tn.set_value("knob.deadband_deg", 6.0)
    exported = tn.export()
    cfg2 = AppConfig()
    tn2 = Tuning(cfg2, path=str(tmp_path / "b.json"))
    applied, rejected = tn2.import_values({**exported, "nope.key": 1})
    assert cfg2.knob.deadband_deg == 6.0 and rejected == {"nope.key": "unknown setting"}


def test_the_live_tuning_file_loads_unchanged(tmp_path):
    """The user's real, hand-tuned file (snapshot 2026-10-30): every value
    applies exactly and a save rewrites it byte-for-byte equivalent."""
    live = {"capture.crop": 1.35, "gate.depth_min_m": 0.3, "ha.max_volume": 0.1,
            "knob.engage_frames": 2, "knob.engage_pinch": 0.6,
            "knob.full_scale_deg": 2100.0, "knob.hand_lost_grace_s": 0.0,
            "knob.release_frames": 2, "playpause.facing_min": 0.5,
            "playpause.max_speed_frac": 0.05, "playpause.min_presence_s": 0.75,
            "swipe.invert": True, "swipe.max_vertical_ratio": 1.1,
            "swipe.min_presence_s": 0.2, "swipe.min_speed_frac": 0.2,
            "swipe.min_travel_frac": 0.11}
    (tmp_path / "tuning.json").write_text(json.dumps(live))
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    for key, value in live.items():
        section, _, attr = key.partition(".")
        assert getattr(getattr(cfg, section), attr) == value, key
    assert tn.export() == live


def test_restoring_twice_does_not_flip_back(tmp_path):
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    tn.set_value("knob.engage_pinch", 0.30)
    tn.reset()
    tn.restore()
    tn.restore()                                 # a second click: same answer
    assert cfg.knob.engage_pinch == 0.30


def test_nan_is_rejected(tmp_path):
    tn = make_tuning(tmp_path)
    with pytest.raises(ValueError):
        tn.set_value("ha.max_volume", float("nan"))


@pytest.mark.parametrize("content", ["", "{trunc", "[]", "null", '"x"'])
def test_a_damaged_file_falls_back_to_the_last_good_copy(tmp_path, content):
    cfg = AppConfig()
    tn = make_tuning(tmp_path, cfg)
    tn.set_value("knob.engage_pinch", 0.30)
    make_tuning(tmp_path, AppConfig())           # a clean boot refreshes lastgood
    (tmp_path / "tuning.json").write_text(content)   # hard freeze mid-save
    cfg2 = AppConfig()
    make_tuning(tmp_path, cfg2)                  # must not raise (boot loop)
    assert cfg2.knob.engage_pinch == 0.30
    assert json.loads((tmp_path / "tuning.json").read_text()) == {"knob.engage_pinch": 0.30}
    assert list(tmp_path.glob("tuning.corrupt-*.json"))   # damaged file kept


def test_a_bad_value_is_skipped_not_fatal(tmp_path):
    (tmp_path / "tuning.json").write_text(json.dumps({"knob.deadband_deg": "abc",
                                                      "gate.min_score": None,
                                                      "knob.engage_frames": 3}))
    cfg = AppConfig()
    make_tuning(tmp_path, cfg)
    assert cfg.knob.engage_frames == 3
