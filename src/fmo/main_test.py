import pytest
from typer.testing import CliRunner
from .main import app, LocalDatasetType, QuantMode

runner = CliRunner()

class TestLocalDatasetType:
    @pytest.mark.parametrize("concrete_type", [
        (LocalDatasetType.JSON, "json"),
        (LocalDatasetType.CSV, "csv"),
        (LocalDatasetType.PARQUET, "parquet"),
        (LocalDatasetType.ARROW, "arrow"),
    ])
    def test_concrete_to_string(self, concrete_type):
        dataset_type, expected_type = concrete_type
        assert dataset_type.to_string("any_path") == expected_type
        
    @pytest.mark.parametrize("valid_extension", [
        ("test.json", "json"),
        ("data.csv", "csv"),
        ("/some/path/data.parquet", "parquet"),
        ("random/path/data.arrow", "arrow"),
    ])
    def test_valid_extensions(self, valid_extension):
        path, expected_type = valid_extension
        assert LocalDatasetType.INFERRED.to_string(path) == expected_type

    @pytest.mark.parametrize("invalid_extension", [
        "",
        "test.txt",
        "data.pdf",
        "model.bin",
        "invalid_path",
    ])
    def test_invalid_extensions(self, invalid_extension):
        with pytest.raises(ValueError, match="Unsupported dataset file extension"):
            LocalDatasetType.INFERRED.to_string(invalid_extension)

class TestCli:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.strip()  # Should contain version string
