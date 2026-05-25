from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceConfig:
    type: str = "langfuse"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    poll_interval_seconds: int = 10
    otlp_endpoint: str = ""
    source_hint: str = ""
    otlp_live_host: str = ""
    otlp_live_port: int = 4318
    otlp_live_auth_token: str = ""
    otlp_live_max_buffer: int = 10000


@dataclass
class Layer1Config:
    stratify_by: list[str] = field(default_factory=lambda: ["agent_version", "task_type", "model_version"])
    min_support: int = 50
    confidence_level: float = 0.95


@dataclass
class Layer2Config:
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache: bool = True
    num_samples: int = 200
    calibration_holdout: float = 0.1


@dataclass
class Layer3Config:
    enabled: bool = False
    replay_mode: str = "recorded_io_replay"
    replay_budget_per_trace: int = 40
    replay_concurrency: int = 4


@dataclass
class Layer4Config:
    enabled: bool = False
    requires_layer_3: bool = True
    claim_type: str = "associational"


@dataclass
class AttributionConfig:
    layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    layer_1: Layer1Config = field(default_factory=Layer1Config)
    layer_2: Layer2Config = field(default_factory=Layer2Config)
    layer_3: Layer3Config = field(default_factory=Layer3Config)
    layer_4: Layer4Config = field(default_factory=Layer4Config)


@dataclass
class PruningConfig:
    prune_epsilon: float = 0.05
    keep_threshold: float = 0.10
    min_trajectories: int = 10
    protect_first_last: bool = True
    validation_gate: bool = True


@dataclass
class OutcomeConfig:
    source: str = "langfuse_score"
    score_name: str = "task_success"
    normalization_baseline: str = "historical_p50"


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    sqlite_path: str = "./traceshap.db"
    retention_days: int = 30


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4


@dataclass
class PerformanceConfig:
    embedding_cache_size: int = 10000
    max_concurrent_analyses: int = 8
    layer_2_timeout_seconds: int = 10


@dataclass
class TraceSHAPConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


def _merge_dataclass(dc_instance, raw_dict: dict):
    if raw_dict is None:
        return dc_instance
    for key, value in raw_dict.items():
        if hasattr(dc_instance, key):
            attr = getattr(dc_instance, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                _merge_dataclass(attr, value)
            else:
                setattr(dc_instance, key, value)
    return dc_instance


def load_config(path: Path) -> TraceSHAPConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    config = TraceSHAPConfig()
    for section_name in ("source", "attribution", "pruning", "outcome", "storage", "server", "performance"):
        if section_name in raw:
            section = getattr(config, section_name)
            _merge_dataclass(section, raw[section_name])
    return config
