# src/mlx_triage/prompts/standard_suite.py
"""Diagnostic prompt suite for Tier 1 statistical tests.

Each prompt targets a specific failure mode identified in the evidence base:
- Math: sensitive to precision drift
- Code: sensitive to constrained decoding, tokenizer issues
- Structured: sensitive to EOS handling, template issues
- Multi-turn: sensitive to context truncation, template rendering
- Edge cases: from production incidents (Anthropic TPU bug, etc.)
"""

from __future__ import annotations

DIAGNOSTIC_PROMPTS: list[dict] = [
    # Math reasoning — sensitive to precision drift
    {
        "id": "math_basic",
        "category": "math",
        "prompt": "What is 2+2? Answer with just the number.",
        "max_tokens": 16,
    },
    {
        "id": "math_chain",
        "category": "math",
        "prompt": "If I have 15 apples and give away 7, then buy 3 more, how many do I have? Answer with just the number.",
        "max_tokens": 16,
    },
    # Code generation — sensitive to constrained decoding, tokenizer issues
    {
        "id": "code_python",
        "category": "code",
        "prompt": "Write a Python function that reverses a string. Only output the code, no explanation.",
        "max_tokens": 128,
    },
    {
        "id": "code_json",
        "category": "code",
        "prompt": 'Output valid JSON with these exact fields: {"name": "test", "value": 42}',
        "max_tokens": 64,
    },
    # Structured output — sensitive to EOS handling, template issues
    {
        "id": "structured_list",
        "category": "structured",
        "prompt": "List exactly 3 fruits, one per line, nothing else.",
        "max_tokens": 32,
    },
    {
        "id": "structured_stop",
        "category": "structured",
        "prompt": "Say 'done' and stop.",
        "max_tokens": 16,
    },
    # Multi-turn — sensitive to context truncation, template rendering
    {
        "id": "multi_turn",
        "category": "multi_turn",
        "prompt": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name?"},
        ],
        "max_tokens": 32,
    },
    # Factual / short-form — baseline quality
    {
        "id": "factual_capital",
        "category": "factual",
        "prompt": "What is the capital of France? Answer with just the city name.",
        "max_tokens": 16,
    },
    # Edge cases from evidence
    {
        "id": "repetition_check",
        "category": "edge",
        "prompt": "Describe the process of photosynthesis in exactly 2 sentences.",
        "max_tokens": 128,
    },
    {
        "id": "unicode_check",
        "category": "edge",
        "prompt": "Translate 'hello' to Japanese. Output only the Japanese text.",
        "max_tokens": 32,
    },
]


def get_prompt(prompt_id: str) -> dict | None:
    """Get a specific diagnostic prompt by ID."""
    for prompt in DIAGNOSTIC_PROMPTS:
        if prompt["id"] == prompt_id:
            return prompt
    return None


def get_all_prompts() -> list[dict]:
    """Get all diagnostic prompts."""
    return list(DIAGNOSTIC_PROMPTS)


# Fixed evaluation corpus for perplexity measurement.
# Public domain text (adapted from Wikipedia-style factual content).
# ~800 tokens when tokenized with typical LLM tokenizers.
EVAL_CORPUS = (
    "The Earth orbits the Sun at an average distance of about 150 million "
    "kilometers. This distance is known as one astronomical unit. The orbital "
    "period is approximately 365.25 days, which is why a leap year occurs every "
    "four years to keep the calendar aligned with the seasons. The Earth's axis "
    "is tilted at approximately 23.5 degrees relative to its orbital plane, "
    "which causes the seasons as different hemispheres receive varying amounts "
    "of solar radiation throughout the year.\n\n"
    "Water covers approximately 71 percent of the Earth's surface, with oceans "
    "holding about 96.5 percent of all water on Earth. The Pacific Ocean is the "
    "largest and deepest ocean, covering more area than all the land masses "
    "combined. The average depth of the ocean is about 3,688 meters, while the "
    "deepest point, the Challenger Deep in the Mariana Trench, reaches "
    "approximately 10,935 meters below sea level.\n\n"
    "The atmosphere of Earth consists primarily of nitrogen at about 78 percent "
    "and oxygen at about 21 percent, with trace amounts of argon, carbon "
    "dioxide, and other gases. The atmosphere extends to about 10,000 kilometers "
    "above the surface, though most of the mass is concentrated in the lowest "
    "layer, the troposphere, which extends to about 12 kilometers altitude. "
    "Weather phenomena occur primarily in the troposphere.\n\n"
    "Photosynthesis is the process by which plants convert light energy into "
    "chemical energy. During photosynthesis, plants absorb carbon dioxide from "
    "the atmosphere and water from the soil, using sunlight as an energy source "
    "to produce glucose and oxygen. The chemical equation for photosynthesis is "
    "6CO2 + 6H2O + light energy produces C6H12O6 + 6O2. This process is "
    "fundamental to life on Earth as it produces the oxygen that most organisms "
    "breathe and forms the base of most food chains.\n\n"
    "Mathematics is the study of numbers, quantities, shapes, and patterns. "
    "The Pythagorean theorem states that in a right triangle, the square of "
    "the hypotenuse equals the sum of the squares of the other two sides. "
    "This can be written as a squared plus b squared equals c squared. "
    "The number pi, approximately equal to 3.14159, represents the ratio of "
    "a circle's circumference to its diameter. Euler's identity, often "
    "considered the most beautiful equation in mathematics, states that e to "
    "the power of i times pi plus one equals zero, connecting five fundamental "
    "mathematical constants."
)
