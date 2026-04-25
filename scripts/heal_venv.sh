#!/usr/bin/env bash
#
# Heal stale editable install state in the local .venv.
#
# When `uv` fails to clean up dist-info directories cleanly — most often
# triggered by version bumps in the local editable package or in
# `solvix-contracts` — the next `uv sync` invocation can fail with:
#
#   error: failed to remove directory `.../<pkg>-X.Y.Z.dist-info`:
#          No such file or directory (os error 2)
#
# This propagates as exit 2 from any `uv run` wrapping pre-commit hook,
# even when the hook script itself would have passed. This helper detects
# the inconsistent state via fast stat checks and re-runs `uv sync`. It's
# idempotent — when the venv is clean, it's a sub-100ms no-op.
#
# Detection heuristics:
#   - any `*.dist-info` directory missing its RECORD file
#   - multiple `*.dist-info` directories for the same package basename
#   - any `_editable_impl_*.pth` whose source directory no longer exists

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_PACKAGES="${REPO_ROOT}/.venv/lib/python3.12/site-packages"

# No venv yet? Nothing to heal.
[[ -d "$SITE_PACKAGES" ]] || exit 0

needs_heal=0

shopt -s nullglob

# (1) Any dist-info missing RECORD = corrupted install.
for d in "$SITE_PACKAGES"/*.dist-info; do
  if [[ ! -f "$d/RECORD" ]]; then
    needs_heal=1
    break
  fi
done

# (2) Multiple dist-info directories for the same package basename = stale state.
# Portable to bash 3.2 (macOS default) — no associative arrays.
if (( ! needs_heal )); then
  duplicate=$(
    for d in "$SITE_PACKAGES"/*.dist-info; do
      base="${d##*/}"
      echo "${base%-*.dist-info}"
    done | sort | uniq -d | head -1
  )
  if [[ -n "$duplicate" ]]; then
    needs_heal=1
  fi
fi

# (3) Editable .pth pointing at a non-existent source dir = orphaned editable install.
if (( ! needs_heal )); then
  for pth in "$SITE_PACKAGES"/_editable_impl_*.pth; do
    [[ -f "$pth" ]] || continue
    src=$(grep -oE 'file://[^"]+' "$pth" 2>/dev/null | head -1 | sed 's|file://||' || true)
    if [[ -n "$src" && ! -d "$src" ]]; then
      needs_heal=1
      break
    fi
  done
fi

shopt -u nullglob

if (( ! needs_heal )); then
  exit 0
fi

echo "→ healing stale editable install state in .venv" >&2

# Wipe known editable install metadata. uv will reinstall fresh on next sync.
rm -rf \
  "$SITE_PACKAGES"/solvix_contracts \
  "$SITE_PACKAGES"/solvix_contracts-*.dist-info \
  "$SITE_PACKAGES"/_editable_impl_solvix_contracts.pth \
  "$SITE_PACKAGES"/outstanding_ai_engine \
  "$SITE_PACKAGES"/outstanding_ai_engine-*.dist-info \
  "$SITE_PACKAGES"/_editable_impl_outstanding_ai_engine.pth

# Re-sync — AI uses [project.optional-dependencies], so --extra dev.
cd "$REPO_ROOT"
uv sync --frozen --extra dev >&2

echo "→ heal complete" >&2
