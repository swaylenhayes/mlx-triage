from unittest.mock import patch

from mlx_triage.models import CheckStatus, TierReport
from mlx_triage.tier0 import run_tier0


def test_run_tier0_returns_report(good_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(good_model))
    assert isinstance(report, TierReport)
    assert report.tier == 0
    assert len(report.checks) == 5


def test_run_tier0_all_pass(good_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(good_model))
    statuses = [c.status for c in report.checks]
    assert CheckStatus.FAIL not in statuses
    assert CheckStatus.CRITICAL not in statuses
    assert report.should_continue is True


def test_run_tier0_catches_vlm_architecture(tmp_path):
    import json

    d = tmp_path / "vlm-model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "vision_config": {"hidden_size": 1024},
                "torch_dtype": "bfloat16",
            }
        )
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "</s>", "chat_template": "..."})
    )
    from tests.conftest import create_safetensors

    create_safetensors(d / "model.safetensors", dtype="BF16")
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(d))
    arch_check = next(c for c in report.checks if c.check_id == "0.5")
    assert arch_check.status == CheckStatus.FAIL
    assert report.should_continue is False


def test_run_tier0_catches_bad_dtype(bf16_to_fp16_model):
    with patch("mlx_triage.tier0.version_check._get_mlx_version", return_value="0.25.1"):
        report = run_tier0(str(bf16_to_fp16_model))
    dtype_check = next(c for c in report.checks if c.check_id == "0.1")
    assert dtype_check.status == CheckStatus.CRITICAL
    assert report.should_continue is False
