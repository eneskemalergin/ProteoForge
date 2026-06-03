#!/usr/bin/env bash
# Dev validation for tools/compare (not part of public tests/ CI).
#
# Usage:
#   tools/compare/validate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== compare validation: parity self-tests =="
PYTHONPATH="${ROOT}" uv run pytest "${ROOT}/tools/compare/validation/test_parity.py" -q

echo
echo "PASS: compare validation"
