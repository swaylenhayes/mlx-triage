"""Check 0.2: Tokenizer & EOS configuration audit.

Evidence basis: Ollama 400+ issues from silent context truncation,
llama.cpp ChatML fallback, vLLM double-BOS injection, Llama 3 dual stop token issue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_triage.models import CheckStatus, DiagnosticResult

# Tokens that indicate Llama 3+ dual-stop-token pattern
LLAMA3_STOP_TOKENS = {"<|end_of_text|>", "<|eot_id|>"}

# Special tokens that indicate thinking/reasoning capability
_THINKING_TOKENS = {"<think>", "</think>"}


def _has_thinking_tokens(tok_config: dict) -> bool:
    """Check if tokenizer vocabulary contains both <think> and </think> special tokens."""
    added_tokens = tok_config.get("added_tokens_decoder", {})
    found: set[str] = set()
    for _id, token_info in added_tokens.items():
        content = token_info.get("content", "")
        if content in _THINKING_TOKENS:
            found.add(content)
    return found == _THINKING_TOKENS


def check_tokenizer_config(model_path: str) -> DiagnosticResult:
    """Run tokenizer and EOS configuration audit on a model directory."""
    path = Path(model_path)
    tokenizer_path = path / "tokenizer_config.json"

    if not tokenizer_path.exists():
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.FAIL,
            detail="tokenizer_config.json not found in model directory.",
        )

    try:
        with open(tokenizer_path) as f:
            tok_config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.FAIL,
            detail=f"Failed to parse tokenizer_config.json: {exc}",
        )

    issues: list[str] = []
    remediations: list[str] = []

    # Check EOS token
    eos_token = tok_config.get("eos_token")
    if not eos_token:
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.CRITICAL,
            detail="No eos_token defined in tokenizer config. Model may never stop generating.",
            remediation="Add eos_token to tokenizer_config.json.",
        )

    # Check chat template
    chat_template = tok_config.get("chat_template")
    metadata: dict[str, Any] = {"has_chat_template": bool(chat_template)}
    metadata["has_thinking_tokens"] = _has_thinking_tokens(tok_config)
    if not chat_template:
        issues.append("No chat_template defined — runtime will fall back to ChatML or default template.")
        remediations.append("Add a chat_template to tokenizer_config.json matching the model's training format.")

    # Check for generation_config.json stop tokens
    gen_config_path = path / "generation_config.json"
    if gen_config_path.exists():
        try:
            with open(gen_config_path) as f:
                gen_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            gen_config = {}

        # Check if multiple stop tokens are needed but not all configured
        eos_token_id = gen_config.get("eos_token_id")
        if isinstance(eos_token_id, int):
            # Single stop token configured — check if model might need multiple
            if isinstance(eos_token, str) and any(t in eos_token for t in LLAMA3_STOP_TOKENS):
                issues.append(
                    f"Llama 3-style EOS token detected ({eos_token}) but generation_config "
                    f"has single eos_token_id. Model may need multiple stop tokens."
                )
                remediations.append("Check if model needs both <|end_of_text|> and <|eot_id|> as stop tokens.")

    if not issues:
        return DiagnosticResult(
            check_id="0.2",
            name="Tokenizer Config",
            status=CheckStatus.PASS,
            detail=f"EOS token: {eos_token}. Chat template: {'present' if chat_template else 'missing'}.",
            metadata=metadata,
        )

    # Determine severity
    has_template_issue = any("chat_template" in i.lower() for i in issues)
    status = CheckStatus.WARNING if has_template_issue else CheckStatus.INFO

    return DiagnosticResult(
        check_id="0.2",
        name="Tokenizer Config",
        status=status,
        detail=" ".join(issues),
        remediation=" ".join(remediations) if remediations else None,
        metadata=metadata,
    )
