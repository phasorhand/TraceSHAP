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

    raise ValueError(f"Unknown source type: '{config.type}'. Supported: langfuse, otlp_json")
