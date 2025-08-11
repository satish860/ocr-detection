# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the OCR Detection Library.

## Workflows

### 1. Tests (`test.yml`)
- **Triggers**: Every push to master/main/develop and all pull requests
- **Actions**: 
  - Runs tests on multiple OS (Ubuntu, Windows, macOS)
  - Tests with Python 3.13 (and 3.10 on Ubuntu for compatibility)
  - Runs linting and type checking (non-blocking)
  - Generates coverage reports
- **Purpose**: Ensures code quality on every commit

### 2. CI/CD Pipeline (`ci-cd.yml`)
- **Triggers**: 
  - Push to main branches
  - Version tags (v*)
  - Release publication
- **Actions**:
  - Runs full test suite
  - Builds distribution packages
  - Publishes to TestPyPI for tags
  - Publishes to PyPI for releases
- **Purpose**: Complete automation from test to deployment

### 3. Release (`release.yml`)
- **Triggers**: 
  - GitHub release creation
  - Manual workflow dispatch
- **Actions**:
  - Runs quality checks
  - Builds and publishes to PyPI
  - Uploads assets to GitHub release
  - Verifies the published package
- **Purpose**: Controlled release process with validation

## Setup Instructions

### 1. Add PyPI API Token to GitHub Secrets

1. Go to your repository on GitHub
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add a secret named `PYPI_API_TOKEN`
5. Paste your PyPI API token as the value

### 2. Creating a Release

#### Option A: Automatic Release via GitHub
1. Go to the Releases page of your repository
2. Click "Create a new release"
3. Create a new tag (e.g., `v0.1.1`)
4. Fill in release notes
5. Click "Publish release"
6. The workflow will automatically build and publish to PyPI

#### Option B: Manual Release via Workflow
1. Go to Actions tab
2. Select "Release to PyPI" workflow
3. Click "Run workflow"
4. Enter the version number
5. Click "Run workflow"

#### Option C: Tag-based Release
```bash
git tag v0.1.1
git push origin v0.1.1
```

## Workflow Features

### Multi-OS Testing
- Tests run on Ubuntu, Windows, and macOS
- Ensures cross-platform compatibility

### Automatic Version Management
- Version tags trigger test releases to TestPyPI
- GitHub releases trigger production PyPI releases

### Quality Gates
- All tests must pass before deployment
- Package validation with twine
- Coverage reporting

### Caching
- UV dependencies are cached for faster builds
- Python setup is cached

## Best Practices

1. **Version Bumping**: Update version in `pyproject.toml` before creating a release
2. **Testing**: Ensure all tests pass locally before pushing
3. **Release Notes**: Write clear release notes for each version
4. **Semantic Versioning**: Follow semantic versioning (MAJOR.MINOR.PATCH)

## Troubleshooting

### Tests Failing on Windows
- Check for path separator issues (use `pathlib.Path`)
- Avoid emoji in code (causes Windows issues)

### PyPI Upload Fails
- Verify `PYPI_API_TOKEN` is set correctly
- Check that the version doesn't already exist on PyPI
- Ensure package metadata is complete

### Workflow Not Triggering
- Check branch protection rules
- Verify workflow file syntax
- Check GitHub Actions is enabled for the repository

## Local Testing

To test the workflows locally:

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Run tests workflow
act -j test

# Run with specific Python version
act -j test -P ubuntu-latest=python:3.13
```

## Security Notes

- Never commit API tokens directly
- Use GitHub Secrets for sensitive data
- Consider using OIDC for PyPI publishing (trusted publishing)
- Review workflow permissions regularly