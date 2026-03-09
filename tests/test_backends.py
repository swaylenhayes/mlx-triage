"""Tests for model backend protocol and factory."""
import json
from unittest.mock import patch

import pytest

from mlx_triage.utils.backends import (
    BackendUnavailable,
    ModelBackend,
    _is_vlm,
    get_backend,
)
from mlx_triage.utils.mlx_utils import MLXLMBackend


class TestVLMDetection:
    def test_detects_vision_config(self):
        assert _is_vlm({"vision_config": {"hidden_size": 1024}}) is True

    def test_detects_visual(self):
        assert _is_vlm({"visual": {"type": "clip"}}) is True

    def test_detects_vision_tower_config(self):
        assert _is_vlm({"vision_tower_config": {}}) is True

    def test_rejects_text_only(self):
        assert _is_vlm({"model_type": "llama"}) is False

    def test_rejects_empty(self):
        assert _is_vlm({}) is False


class TestGetBackend:
    def test_returns_mlxlm_for_text_model(self, good_model):
        backend = get_backend(str(good_model))
        assert isinstance(backend, MLXLMBackend)

    def test_raises_for_vlm(self, tmp_path):
        d = tmp_path / "vlm"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"vision_config": {"hidden_size": 1024}})
        )
        with pytest.raises(BackendUnavailable, match="VLM model detected"):
            get_backend(str(d))

    def test_handles_missing_config(self, tmp_path):
        d = tmp_path / "bare"
        d.mkdir()
        backend = get_backend(str(d))
        assert isinstance(backend, MLXLMBackend)

    def test_handles_corrupt_config(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "config.json").write_text("not json")
        backend = get_backend(str(d))
        assert isinstance(backend, MLXLMBackend)


class TestMLXLMBackend:
    def test_satisfies_protocol(self):
        assert isinstance(MLXLMBackend(), ModelBackend)

    def test_is_available_delegates(self):
        backend = MLXLMBackend()
        with patch(
            "mlx_triage.utils.mlx_utils.check_mlx_available", return_value=True
        ):
            assert backend.is_available() is True

    def test_is_not_available(self):
        backend = MLXLMBackend()
        with patch(
            "mlx_triage.utils.mlx_utils.check_mlx_available", return_value=False
        ):
            assert backend.is_available() is False


class TestBackendUnavailable:
    def test_is_exception(self):
        with pytest.raises(BackendUnavailable):
            raise BackendUnavailable("test")

    def test_message(self):
        exc = BackendUnavailable("VLM not supported")
        assert "VLM" in str(exc)
