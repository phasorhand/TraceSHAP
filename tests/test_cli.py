import pytest
from click.testing import CliRunner

from traceshap.cli.main import cli


class TestCLIEntryPoint:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "TraceSHAP" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_init_creates_config(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        config_path = tmp_path / "traceshap.yaml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "source:" in content
        assert "attribution:" in content
