# Dev-only validation for reference vs ProteoForge compare tooling

This folder holds **local development checks** that depend on reference
parity logic or compare harness behavior. It is intentionally **not** part of
the public `tests/` suite run in CI.

## Contents

| File | Purpose |
| ---- | ------- |
| `test_parity.py` | Self-tests for `compare_parity()` (key alignment, anti-cheese cases) |

Run via:

```bash
tools/compare/validate.sh
```

Live reference vs ProteoForge checks use `tools/compare/zebrac.sh` and require
local `ref/` + fixtures (gitignored).
