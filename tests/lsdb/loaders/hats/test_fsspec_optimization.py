"""Tests for fsspec optimization functionality."""

import os
from unittest.mock import patch

import pytest

from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig


class TestFsspecOptimization:
    """Test fsspec optimization parameter handling."""

    def test_explicit_enable(self):
        """Test explicit enable of fsspec optimization."""
        config = HatsLoadingConfig(enable_fsspec_optimization=True)
        kwargs = config.get_read_kwargs()

        assert "open_file_options" in kwargs
        assert "precache_options" in kwargs["open_file_options"]
        assert kwargs["open_file_options"]["precache_options"]["method"] == "parquet"

    def test_explicit_disable(self):
        """Test explicit disable of fsspec optimization."""
        config = HatsLoadingConfig(enable_fsspec_optimization=False)
        kwargs = config.get_read_kwargs()

        # Should not add open_file_options if not already present
        assert "open_file_options" not in kwargs or "precache_options" not in kwargs.get(
            "open_file_options", {}
        )

    def test_environment_variable_true(self):
        """Test environment variable set to true."""
        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "true"}):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is True

            kwargs = config.get_read_kwargs()
            assert "open_file_options" in kwargs
            assert kwargs["open_file_options"]["precache_options"]["method"] == "parquet"

    def test_environment_variable_false(self):
        """Test environment variable set to false."""
        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "false"}):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is False

    def test_environment_variable_numeric(self):
        """Test environment variable with numeric values."""
        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "1"}):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is True

        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "0"}):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is False

    def test_environment_variable_case_insensitive(self):
        """Test environment variable is case insensitive."""
        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "TRUE"}):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is True

        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "FALSE"}):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is False

    def test_environment_variable_other_values(self):
        """Test environment variable with other true/false values."""
        for true_value in ["yes", "on", "YES", "ON"]:
            with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": true_value}):
                config = HatsLoadingConfig()
                assert config.enable_fsspec_optimization is True

    def test_no_environment_variable(self):
        """Test default behavior when no environment variable is set."""
        with patch.dict(os.environ, {}, clear=True):
            config = HatsLoadingConfig()
            assert config.enable_fsspec_optimization is False

    def test_explicit_overrides_environment(self):
        """Test that explicit parameter overrides environment variable."""
        with patch.dict(os.environ, {"LSDB_ENABLE_FSSPEC_OPTIMIZATION": "true"}):
            config = HatsLoadingConfig(enable_fsspec_optimization=False)
            assert config.enable_fsspec_optimization is False

    def test_preserve_user_precache_options(self):
        """Test that user-provided precache_options are not overridden."""
        user_kwargs = {"open_file_options": {"precache_options": {"method": "custom", "size": 1024}}}
        config = HatsLoadingConfig(enable_fsspec_optimization=True, kwargs=user_kwargs)
        kwargs = config.get_read_kwargs()

        # Should not override user's precache_options
        assert kwargs["open_file_options"]["precache_options"]["method"] == "custom"
        assert kwargs["open_file_options"]["precache_options"]["size"] == 1024

    def test_preserve_other_open_file_options(self):
        """Test that other open_file_options are preserved when adding precache_options."""
        user_kwargs = {"open_file_options": {"block_size": 1024, "cache_type": "bytes"}}
        config = HatsLoadingConfig(enable_fsspec_optimization=True, kwargs=user_kwargs)
        kwargs = config.get_read_kwargs()

        # Should add precache_options but preserve other options
        assert kwargs["open_file_options"]["precache_options"]["method"] == "parquet"
        assert kwargs["open_file_options"]["block_size"] == 1024
        assert kwargs["open_file_options"]["cache_type"] == "bytes"

    def test_preserve_other_kwargs(self):
        """Test that other kwargs are preserved."""
        user_kwargs = {"filters": [("column", "==", "value")], "columns": ["col1", "col2"]}
        config = HatsLoadingConfig(enable_fsspec_optimization=True, kwargs=user_kwargs)
        kwargs = config.get_read_kwargs()

        # Should preserve other kwargs
        assert kwargs["filters"] == [("column", "==", "value")]
        assert kwargs["columns"] == ["col1", "col2"]
        assert kwargs["open_file_options"]["precache_options"]["method"] == "parquet"
