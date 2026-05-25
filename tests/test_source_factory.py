import pytest
import json

from traceshap.config import SourceConfig
from traceshap.ingestion.sources.factory import create_source
from traceshap.ingestion.sources.langfuse import LangfuseSource
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource
from traceshap.ingestion.sources.otlp_live import OTLPLiveSource
from traceshap.ingestion.sources.hermes_jsonl import HermesJsonlSource


class TestSourceFactory:
    def test_create_langfuse_source(self):
        config = SourceConfig(
            type="langfuse",
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        )
        source = create_source(config)
        assert isinstance(source, LangfuseSource)

    def test_create_otlp_json_source(self, tmp_path):
        data = {"resourceSpans": []}
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        config = SourceConfig(type="otlp_json", otlp_endpoint=str(path))
        source = create_source(config)
        assert isinstance(source, OTLPJsonSource)

    def test_unknown_type_raises(self):
        config = SourceConfig(type="unknown_source")
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source(config)

    def test_otlp_json_with_source_hint(self, tmp_path):
        data = {"resourceSpans": []}
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(data))

        config = SourceConfig(type="otlp_json", otlp_endpoint=str(path), source_hint="openclaw")
        source = create_source(config)
        assert isinstance(source, OTLPJsonSource)

    def test_create_otlp_live_source(self):
        config = SourceConfig(
            type="otlp_live",
            otlp_live_host="127.0.0.1",
            otlp_live_port=4318,
            otlp_live_auth_token="secret-token",
            otlp_live_max_buffer=5000,
        )
        source = create_source(config)
        assert isinstance(source, OTLPLiveSource)

    def test_create_hermes_jsonl_source(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        path.write_text('{"spans": []}\n')

        config = SourceConfig(type="hermes_jsonl", otlp_endpoint=str(path))
        source = create_source(config)
        assert isinstance(source, HermesJsonlSource)

    def test_create_unknown_source_updated_error(self):
        config = SourceConfig(type="magic")
        with pytest.raises(ValueError, match="hermes_jsonl") as exc_info:
            create_source(config)
        assert "otlp_live" in str(exc_info.value)
