import pytest
import yaml

from ted_curation.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config


def test_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg["paths"]["raw_dir"] == PROJECT_ROOT / "data" / "raw"
    assert isinstance(cfg["project"]["seed"], int)


def test_invalid_split_ratios_raise(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["splits"]["train"] = 0.9  # now sums to 1.2
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(bad)
