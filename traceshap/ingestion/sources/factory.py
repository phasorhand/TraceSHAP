from traceshap.config import SourceConfig
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.sources.langfuse import LangfuseSource
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource


def create_source(config: SourceConfig) -> SpanSource:
    if config.type == "langfuse":
        return LangfuseSource(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
            poll_batch_size=config.poll_interval_seconds,
        )

    if config.type == "otlp_json":
        return OTLPJsonSource(
            path=config.otlp_endpoint,
            source_hint=config.source_hint or "otlp",
        )

    if config.type == "otlp_live":
        from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
        return OTLPLiveSource(
            host=config.otlp_live_host or "0.0.0.0",
            port=config.otlp_live_port or 4318,
            source_hint=config.source_hint or "otlp",
            auth_token=config.otlp_live_auth_token or None,
            max_buffer_size=config.otlp_live_max_buffer or 10000,
        )

    if config.type == "hermes_jsonl":
        from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource
        return HermesJsonlSource(path=config.otlp_endpoint)

    raise ValueError(f"Unknown source type: '{config.type}'. Supported: langfuse, otlp_json, otlp_live, hermes_jsonl")
