# Makefile for OCR Detection Library
# Run 'make help' to see available commands

.PHONY: help install test lint format clean build publish

help:  ## Show this help message
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync --dev
	uv run pre-commit install

test:  ## Run all tests
	uv run pytest tests/ -v

test-cov:  ## Run tests with coverage
	uv run pytest tests/ --cov=ocr_detection --cov-report=term-missing --cov-report=html

lint:  ## Run linting checks (ruff, mypy)
	uv run ruff check src/ tests/
	uv run mypy src/

lint-fix:  ## Auto-fix linting issues
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

format:  ## Format code with black and ruff
	uv run black src/ tests/
	uv run ruff format src/ tests/

type-check:  ## Run type checking with mypy
	uv run mypy src/

security:  ## Run security checks with bandit
	uv add --dev bandit[toml]
	uv run bandit -r src/ -ll

pre-commit:  ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

clean:  ## Clean build artifacts and cache
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	uv build

check-build:  ## Check distribution packages
	uv run twine check dist/*

publish-test:  ## Publish to TestPyPI
	uv run twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI (requires PYPI_API_TOKEN)
	uv run twine upload dist/*

dev:  ## Run development server/watcher (if applicable)
	@echo "No development server configured"

all: clean lint test build  ## Run all checks and build

ci: lint test  ## Run CI checks locally

quality: lint type-check security test-cov  ## Run all quality checks

# Windows-compatible commands (use these on Windows)
ifeq ($(OS),Windows_NT)
clean-win:  ## Clean build artifacts (Windows)
	if exist dist rmdir /s /q dist
	if exist build rmdir /s /q build
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	if exist .mypy_cache rmdir /s /q .mypy_cache
	if exist htmlcov rmdir /s /q htmlcov
	if exist .coverage del .coverage
	if exist coverage.xml del coverage.xml
endif