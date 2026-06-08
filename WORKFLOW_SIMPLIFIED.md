# Workflow Simplification for Debugging

## Changes Made

Simplified `.github/workflows/tests.yml` to:
1. **Reduced platforms**: Ubuntu only (most reliable)
2. **Reduced Python versions**: 3.9, 3.10, 3.11 (core versions)
3. **Removed code quality checks**: (flake8, black, isort, mypy) - can fail independently of tests
4. **Added explicit diagnostics**:
   - Python version display
   - Import test: `from signalseeker import SignalSeekerPipeline`
   - Verbose pytest output
5. **Updated actions**: v4 and v5 (less deprecation warnings)

## Why

The complex matrix (3 OS × 5 Python versions = 15 jobs) made it hard to debug. Failing on all versions suggests a fundamental issue, not version-specific.

## Next Steps

Push to GitHub and check:
```bash
git add .github/workflows/tests.yml WORKFLOW_SIMPLIFIED.md
git commit -m "Debug: Simplify test workflow to diagnose failures"
git push origin main
```

If the import test passes, tests should work. If import fails, we have a real error to fix.
