import pytest
from pathlib import Path
from traceshap.config import TraceSHAPConfig, load_config


MINIMAL_YAML = """\
source:
  type: langfuse
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-test
  langfuse_secret_key: sk-test

storage:
  backend: sqlite
  sqlite_path: ./test.db
"""

FULL_YAML = """\
source:
  type: langfuse
  langfuse_host: https://cloud.langfuse.com
  langfuse_public_key: pk-test
  langfuse_secret_key: sk-test
  poll_interval_seconds: 5

attribution:
  layers: [0, 1, 2]
  layer_1:
    min_support: 30
  layer_2:
    num_samples: 100

pruning:
  prune_epsilon: 0.03
  keep_threshold: 0.15
  min_trajectories: 20

storage:
  backend: sqlite
  sqlite_path: ./test.db
  retention_days: 14

server:
  host: 127.0.0.1
  port: 9090
"""


class TestLoadConfig:
    def test_minimal_config(self, tmp_path: Path):
        cfg_path = tmp_path / "traceshap.yaml"
        cfg_path.write_text(MINIMAL_YAML)
        config = load_config(cfg_path)
        assert config.source.type == "langfuse"
        assert config.source.langfuse_host == "https://cloud.langfuse.com"
        assert config.storage.backend == "sqlite"
        # defaults
        assert config.source.poll_interval_seconds == 10
        assert config.attribution.layers == [0, 1, 2]
        assert config.server.port == 8080

    def test_full_config(self, tmp_path: Path):
        cfg_path = tmp_path / "traceshap.yaml"
        cfg_path.write_text(FULL_YAML)
        config = load_config(cfg_path)
        assert config.source.poll_interval_seconds == 5
        assert config.attribution.layers == [0, 1, 2]
        assert config.pruning.prune_epsilon == 0.03
        assert config.storage.retention_days == 14
        assert config.server.port == 9090

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/traceshap.yaml"))
