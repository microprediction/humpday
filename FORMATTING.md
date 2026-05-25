# Formatting Reconciliation Guide

This guide explains how to reconcile local formatting/linting with GitHub CI to avoid the dreaded "All checks passed! Would reformat: ..." error.

## The Problem

GitHub CI and local environments can have different versions of formatting tools (ruff, black, etc.), leading to inconsistent formatting results. This causes frustrating failures where local tests pass but GitHub CI fails with formatting violations.

## The Solution: Virtual Environment Approach

Use a temporary virtual environment with the exact same tool versions as GitHub CI.

### Step-by-Step Instructions

1. **Create temporary virtual environment:**
   ```bash
   python3 -m venv /tmp/lint_env
   source /tmp/lint_env/bin/activate
   ```

2. **Install exact same tools as GitHub CI:**
   ```bash
   pip install ruff
   # Add other formatting tools used by your project
   ```

3. **Run exact same commands as GitHub CI:**
   ```bash
   # Check formatting (don't fix yet)
   ruff format --check .
   
   # Check linting
   ruff check .
   ```

4. **If formatting needed, apply it:**
   ```bash
   ruff format .
   ```

5. **Clean up:**
   ```bash
   deactivate
   rm -rf /tmp/lint_env
   ```

### For Other Projects

1. **Check your GitHub CI workflow** (usually `.github/workflows/`) to see exact commands used
2. **Use the same tool versions** as specified in the workflow
3. **Run commands in the same order** as the CI pipeline

### Key Principles

- **Use isolated environments**: Don't rely on system-wide tool installations
- **Match CI exactly**: Same tools, same versions, same commands
- **Check before fixing**: Always run with `--check` first to see what needs formatting
- **Separate formatting from linting**: Handle them as separate steps

### Example GitHub CI Commands to Match

If your `.github/workflows/` file has:
```yaml
- name: Check formatting
  run: ruff format --check .
- name: Check linting  
  run: ruff check .
```

Then locally use:
```bash
python3 -m venv /tmp/lint_env
source /tmp/lint_env/bin/activate
pip install ruff
ruff format --check .
ruff check .
# If needed:
ruff format .
deactivate
rm -rf /tmp/lint_env
```

This approach ensures 100% consistency between local and CI environments, eliminating formatting-related CI failures.