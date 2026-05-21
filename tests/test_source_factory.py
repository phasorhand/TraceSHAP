import pytest
import json

from traceshap.config import SourceConfig
from traceshap.ingestion.sources.factory import create_source
from traceshap.ingestion.sources.langfuse import LangfuseSource
from traceshap.ingestion.sources.otlp_json import OTLPJsonSource


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
