#!/usr/bin/env bash
# scripts/sync-mlx-docs.sh
# Download MLX and mlx-lm documentation into a local searchable corpus.
#
# Usage:
#   ./scripts/sync-mlx-docs.sh          # Sync if stale (>7 days)
#   ./scripts/sync-mlx-docs.sh --force  # Force re-sync
#
# Corpus is stored in _docs/mlx-knowledge/ (gitignored).
# Search with: grep -r "pattern" _docs/mlx-knowledge/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_DIR="$PROJECT_ROOT/_docs/mlx-knowledge"
STALE_DAYS=7

# --- Freshness check ---
if [ -f "$CORPUS_DIR/.last-updated" ] && [ "${1:-}" != "--force" ]; then
    if [ -n "$(find "$CORPUS_DIR/.last-updated" -mtime -${STALE_DAYS} 2>/dev/null)" ]; then
        echo "Corpus is fresh (< ${STALE_DAYS} days). Use --force to re-sync."
        exit 0
    fi
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "=== Syncing MLX documentation corpus ==="

# --- MLX Core Docs (RST source files) ---
echo "[1/3] Cloning ml-explore/mlx docs..."
git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/ml-explore/mlx.git "$TEMP_DIR/mlx" 2>&1 | tail -1
(cd "$TEMP_DIR/mlx" && git sparse-checkout set docs/src/python docs/src/usage docs/src/examples)

mkdir -p "$CORPUS_DIR/mlx-core"
cp -r "$TEMP_DIR/mlx/docs/src/python" "$CORPUS_DIR/mlx-core/"
cp -r "$TEMP_DIR/mlx/docs/src/usage" "$CORPUS_DIR/mlx-core/"
[ -d "$TEMP_DIR/mlx/docs/src/examples" ] && cp -r "$TEMP_DIR/mlx/docs/src/examples" "$CORPUS_DIR/mlx-core/" || true

# Capture MLX version from setup
MLX_VERSION=$(grep -m1 'version' "$TEMP_DIR/mlx/python/setup.py" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")

# --- mlx-lm Docs + Key Source Files ---
echo "[2/3] Cloning ml-explore/mlx-lm docs and source..."
git clone --depth=1 https://github.com/ml-explore/mlx-lm.git "$TEMP_DIR/mlx-lm" 2>&1 | tail -1

mkdir -p "$CORPUS_DIR/mlx-lm/docs" "$CORPUS_DIR/mlx-lm/source"

# Markdown docs
for f in README.md BENCHMARKS.md SERVER.md LORA.md MANAGE.md LEARNED_QUANTS.md; do
    [ -f "$TEMP_DIR/mlx-lm/mlx_lm/$f" ] && cp "$TEMP_DIR/mlx-lm/mlx_lm/$f" "$CORPUS_DIR/mlx-lm/docs/"
done

# Key source files (for API reference via docstrings)
for f in __init__.py generate.py utils.py sample_utils.py perplexity.py evaluate.py; do
    [ -f "$TEMP_DIR/mlx-lm/mlx_lm/$f" ] && cp "$TEMP_DIR/mlx-lm/mlx_lm/$f" "$CORPUS_DIR/mlx-lm/source/"
done

MLXLM_VERSION=$(grep -m1 'version' "$TEMP_DIR/mlx-lm/setup.py" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")

# --- Metadata ---
echo "[3/3] Writing metadata..."
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$CORPUS_DIR/.last-updated"

cat > "$CORPUS_DIR/.sync-info" <<EOF
MLX Core: $MLX_VERSION (from ml-explore/mlx main branch)
mlx-lm: $MLXLM_VERSION (from ml-explore/mlx-lm main branch)
Synced: $(cat "$CORPUS_DIR/.last-updated")
EOF

echo "=== Done ==="
echo "Corpus: $CORPUS_DIR"
echo "MLX: $MLX_VERSION | mlx-lm: $MLXLM_VERSION"
echo "Search: grep -r 'pattern' $CORPUS_DIR/"
