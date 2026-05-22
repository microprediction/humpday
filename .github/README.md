# CI/CD Setup for Humpday

This directory contains GitHub Actions workflows for automated testing, quality checks, and deployment.

## Workflows

### 🔄 `ci.yml` - Continuous Integration
**Triggers:** Push to main, PRs
**Purpose:** Core testing across platforms and Python versions

- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Code quality**: Ruff linting and formatting
- **Type checking**: MyPy (non-blocking)
- **Test coverage**: pytest with coverage reporting
- **Minimal dependency test**: Ensures core functionality works with minimal deps

### 📦 `publish.yml` - PyPI Publishing
**Triggers:** GitHub releases, manual dispatch
**Purpose:** Build and publish packages to PyPI

- **Safe build process**: Separate build, test, and publish steps
- **Trusted publishing**: Uses OIDC instead of API tokens
- **Test PyPI support**: Manual option to deploy to Test PyPI first
- **Wheel validation**: Tests installation before publishing

### 🔍 `quality.yml` - Quality Assurance
**Triggers:** Push to main, PRs, weekly schedule
**Purpose:** Additional quality and security checks

- **Security scanning**: Safety and Bandit checks
- **Compatibility testing**: Core vs extra dependencies
- **Performance benchmarks**: Basic optimization timing
- **Documentation validation**: README and metadata checks

### Legacy Workflows (to be removed)
- `tests.yml` - Replaced by `ci.yml`
- `deploy.yml` - Replaced by `publish.yml`
- `test-optuna.yml` - Specific testing, may keep or integrate

## Deployment Process

### Automatic (Recommended)
1. Update version in `pyproject.toml`
2. Commit and push changes
3. Create GitHub release with tag `v{version}`
4. Workflows automatically build and publish to PyPI

### Manual Release Helper
```bash
# Update version and run checks
python scripts/release.py --version 0.8.1 --type patch

# Review and commit
git add -A && git commit -m "Release 0.8.1"
git tag v0.8.1
git push && git push --tags

# Create GitHub release (triggers deployment)
```

### Test Deployment
```bash
# Deploy to Test PyPI first
gh workflow run publish.yml --ref main -f test_pypi=true

# Test installation
pip install --index-url https://test.pypi.org/simple/ humpday==0.8.1
```

## Setup Requirements

### Repository Secrets (Not needed with trusted publishing)
Modern workflow uses OIDC trusted publishing - no secrets required!

### Environment Protection (Optional)
Consider setting up environment protection rules:
- `pypi` environment: Require review for production releases
- `test-pypi` environment: Allow automatic deployment

### Branch Protection
Recommended branch protection rules for `main`:
- Require PR reviews
- Require status checks (CI workflow)
- No direct pushes to main

## Troubleshooting

### Coverage Issues
If coverage reporting fails due to numpy import issues:
- Check Python version compatibility
- Review test isolation
- Consider using `--cov-report=term` only

### Build Failures
- Ensure `pyproject.toml` is valid
- Check Python version constraints
- Verify all dependencies are available

### Deployment Issues
- Confirm GitHub release was created (not just tag)
- Check workflow permissions
- Verify PyPI trusted publishing is configured

## Migration from Old Workflows

The new workflows replace:
- `tests.yml` → `ci.yml` (better Python version matrix)
- `deploy.yml` → `publish.yml` (modern build tools, trusted publishing)

Old workflows can be removed after confirming new ones work correctly.