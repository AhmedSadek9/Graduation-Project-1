
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "ocr_config.json"


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_config_is_valid_json():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert isinstance(config, dict)


def test_config_has_required_keys():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert "engine" in config
    assert "preprocess" in config
    assert "paddle" in config
    assert "tesseract" in config


def test_config_engine_is_valid():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert config["engine"] in ["paddle", "tesseract"]


def test_config_preprocess_is_boolean():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert isinstance(config["preprocess"], bool)
