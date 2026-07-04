import os
from unittest import mock

from kinectknob.config import AppConfig, load_config


def test_defaults():
    cfg = load_config(None)
    assert cfg.capture.backend == "auto"
    assert cfg.knob.full_scale_deg == 270.0
    assert cfg.web.port == 8420
    assert cfg.capture.mirror is True


def test_yaml_and_env_precedence(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "ha:\n  url: http://yaml:8123\n  volume_entity: media_player.from_yaml\n"
        "knob:\n  full_scale_deg: 300\n  invert: true\n"
        "web:\n  port: 9000\n"
    )
    env = {"KK_HA_URL": "http://env:8123", "KK_PORT": "9999", "KK_INVERT_ROTATION": "false"}
    with mock.patch.dict(os.environ, env, clear=False):
        cfg = load_config(str(yaml_file))
    assert cfg.ha.url == "http://env:8123"            # env beats yaml
    assert cfg.ha.volume_entity == "media_player.from_yaml"
    assert cfg.knob.full_scale_deg == 300.0
    assert cfg.knob.invert is False                   # env override of yaml bool
    assert cfg.web.port == 9999


def test_media_entity_defaults_to_volume_entity():
    with mock.patch.dict(os.environ, {"KK_VOLUME_ENTITY": "media_player.bose"}, clear=False):
        cfg = load_config(None)
    assert cfg.ha.media_entity == "media_player.bose"


def test_separate_media_entity():
    env = {
        "KK_VOLUME_ENTITY": "media_player.bose",
        "KK_MEDIA_ENTITY": "media_player.spotify_yash",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        cfg = load_config(None)
    assert cfg.ha.volume_entity == "media_player.bose"
    assert cfg.ha.media_entity == "media_player.spotify_yash"


def test_bool_env_parsing():
    for raw, expected in [("1", True), ("true", True), ("YES", True), ("0", False), ("off", False)]:
        with mock.patch.dict(os.environ, {"KK_MIRROR": raw}, clear=False):
            assert load_config(None).capture.mirror is expected, raw


def test_max_volume_clamped():
    with mock.patch.dict(os.environ, {"KK_MAX_VOLUME": "1.7"}, clear=False):
        assert load_config(None).ha.max_volume == 1.0


def test_unknown_yaml_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("knob:\n  full_scale_degrees: 300\n")  # typo'd key
    try:
        load_config(str(bad))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_appconfig_is_self_contained():
    cfg = AppConfig()
    assert cfg.ha.media_entity == ""  # only load_config applies the fallback
