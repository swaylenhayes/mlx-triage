"""Tests for Check 0.4: MLX Version & Known Bug Check."""

import json
from unittest.mock import patch

from mlx_triage.models import CheckStatus
from mlx_triage.tier0.version_check import check_mlx_version


_VERSION_PATCH = "mlx_triage.tier0.version_check._get_mlx_version"


def test_current_version_llama_no_critical(good_model):
    """Current MLX (0.25.1) + llama should not trigger version-specific bugs.

    Only 'all'-version bugs (e.g. MLX-003, MLX-004) may match, so the
    result should be PASS, INFO, or WARNING — never CRITICAL.
    """
    with patch(_VERSION_PATCH, return_value="0.25.1"):
        result = check_mlx_version(str(good_model))

    assert result.status in (CheckStatus.PASS, CheckStatus.INFO, CheckStatus.WARNING)
    assert result.status != CheckStatus.CRITICAL
    assert result.metadata.get("mlx_version") == "0.25.1"
    assert result.metadata.get("architecture") == "llama"


def test_old_version_llama_critical(good_model):
    """Old MLX (0.21.0) + llama should match MLX-001 (addmm bug, critical)."""
    with patch(_VERSION_PATCH, return_value="0.21.0"):
        result = check_mlx_version(str(good_model))

    assert result.status in (CheckStatus.CRITICAL, CheckStatus.WARNING)
    matched_ids = result.metadata.get("matched_bug_ids", [])
    assert "MLX-001" in matched_ids


def test_mlx_not_installed(good_model):
    """When MLX is not installed, the check should SKIP."""
    with patch(_VERSION_PATCH, return_value=None):
        result = check_mlx_version(str(good_model))

    assert result.status == CheckStatus.SKIP
    assert "Install MLX" in (result.remediation or "")


def test_whisper_matches_conv1d_drift(tmp_path):
    """Whisper architecture should match MLX-005 (Conv1d drift, affected_versions: all)."""
    model_dir = tmp_path / "whisper-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "whisper", "torch_dtype": "float32"})
    )

    with patch(_VERSION_PATCH, return_value="0.25.1"):
        result = check_mlx_version(str(model_dir))

    matched_ids = result.metadata.get("matched_bug_ids", [])
    assert "MLX-005" in matched_ids


def test_old_version_llama_matches_mlx002(good_model):
    """Old MLX (0.21.0) + llama should also match MLX-002 (qmv kernel bug)."""
    with patch(_VERSION_PATCH, return_value="0.21.0"):
        result = check_mlx_version(str(good_model))

    matched_ids = result.metadata.get("matched_bug_ids", [])
    assert "MLX-002" in matched_ids


def test_missing_config_json(tmp_path):
    """Model directory without config.json should SKIP."""
    empty_dir = tmp_path / "no-config"
    empty_dir.mkdir()

    with patch(_VERSION_PATCH, return_value="0.25.1"):
        result = check_mlx_version(str(empty_dir))

    assert result.status == CheckStatus.SKIP
    assert result.metadata.get("mlx_version") == "0.25.1"


def test_result_has_remediation_when_bugs_match(good_model):
    """When bugs match, remediation text should reference them."""
    with patch(_VERSION_PATCH, return_value="0.21.0"):
        result = check_mlx_version(str(good_model))

    assert result.remediation is not None
    assert "MLX-001" in result.remediation
