# Contributing to OCR Detection Library

Thank you for your interest in contributing to the OCR Detection Library! We welcome contributions from the community and are grateful for any help you can provide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in your communications
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/ocr-detection.git
   cd ocr-detection
   ```
3. **Add the upstream repository** as a remote:
   ```bash
   git remote add upstream https://github.com/satish860/ocr-detection.git
   ```
4. **Keep your fork up to date**:
   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

## How to Contribute

### Finding Ways to Contribute

- Look for issues labeled `good first issue` or `help wanted`
- Review and improve documentation
- Add or improve tests
- Report bugs or suggest features
- Help other users with their questions

### Before You Start

1. Check if an issue already exists for your contribution
2. For significant changes, open an issue first to discuss your proposal
3. Ensure you have read and understood the project architecture in CLAUDE.md

## Development Setup

### Prerequisites

- Python 3.13 or higher
- UV package manager (recommended) or pip
- Git

### Setting Up Your Development Environment

```bash
# Install UV if you haven't already
# See: https://github.com/astral-sh/uv

# Install dependencies
uv sync

# Verify installation
uv run ocr-detect --version
```

### Project Structure

```
ocr-detection/
├── src/ocr_detection/      # Main package
│   ├── detector.py         # Core PDF analyzer
│   ├── analyzer.py         # Content analysis
│   ├── simple.py          # Simple API
│   ├── api.py             # Enhanced API
│   └── cli.py             # CLI interface
├── tests/                  # Test suite
├── CLAUDE.md              # Project documentation for AI assistants
├── README.md              # User documentation
└── pyproject.toml         # Project configuration
```

## Testing

### Running Tests

Always run tests before submitting a pull request:

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_detector.py

# Run specific test class
uv run pytest tests/test_detector.py::TestParallelProcessing

# Run with coverage
uv run pytest tests/ --cov=ocr_detection

# Run basic functionality test
uv run python test_basic.py
```

### Writing Tests

- Add tests for any new functionality
- Ensure tests are descriptive and well-documented
- Follow the existing test structure and naming conventions
- Aim for high test coverage (>80%)

## Pull Request Process

### Creating a Pull Request

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Add or update tests as needed
   - Update documentation if necessary

3. **Test your changes**:
   ```bash
   # Run tests
   uv run pytest tests/
   
   # Test CLI functionality
   uv run ocr-detect sample.pdf --verbose
   ```

4. **Commit your changes** (see [Commit Message Guidelines](#commit-message-guidelines))

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include any relevant testing information

### Pull Request Review Process

1. Automated tests will run on your PR
2. A maintainer will review your code
3. Address any feedback or requested changes
4. Once approved, your PR will be merged

### Pull Request Checklist

- [ ] Tests pass locally (`uv run pytest tests/`)
- [ ] Code follows the project's style guidelines
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and follow guidelines
- [ ] PR description clearly explains the changes
- [ ] Any new dependencies are justified and documented

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 conventions
- Use meaningful variable and function names
- Add type hints where appropriate
- Keep functions focused and concise
- Document complex logic with comments

### Key Conventions

1. **Class Structure**: Follow the existing patterns in `detector.py` and `analyzer.py`
2. **Error Handling**: Use try-except blocks appropriately, avoid bare exceptions
3. **Logging**: Use the existing logging patterns, avoid print statements
4. **Threading**: Ensure thread-safety when working with parallel processing
5. **No Emojis**: Do not use emojis in code (causes issues on Windows)

### Example Code Style

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """Result of PDF page analysis."""
    page_num: int
    page_type: PageType
    confidence: float
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if analysis has high confidence.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            True if confidence exceeds threshold
        """
        return self.confidence >= threshold
```

## Commit Message Guidelines

### Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Examples

```
feat: Add batch processing support for multiple PDFs

- Implement BatchProcessor class
- Add CLI support for directory input
- Include progress bar for large batches

Closes #123
```

```
fix: Correct confidence calculation for mixed pages

The confidence score was incorrectly weighted for pages
with both text and images. This fix properly balances
the scoring based on content ratios.
```

## Reporting Issues

### Before Reporting

1. Check existing issues to avoid duplicates
2. Verify the issue with the latest version
3. Collect relevant information about your environment

### Issue Template

When reporting issues, please include:

- **Description**: Clear explanation of the issue
- **Steps to Reproduce**: Detailed steps to recreate the problem
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**:
  - OS and version
  - Python version
  - Library version (`uv run ocr-detect --version`)
- **Sample PDF**: If possible, provide a sample PDF that demonstrates the issue

## Feature Requests

### Suggesting Features

1. Check if the feature has already been requested
2. Open an issue with the `feature request` label
3. Provide:
   - Clear description of the feature
   - Use cases and benefits
   - Potential implementation approach (optional)
   - Examples from other tools (if applicable)

### Feature Discussion

- Be open to feedback and alternative approaches
- Help refine the feature specification
- Consider contributing the implementation yourself

## Questions and Support

- For questions about usage, check the README and documentation first
- For development questions, open a discussion or issue
- For general support, use the issue tracker with the `question` label

## Recognition

Contributors will be recognized in the project's release notes and documentation. We appreciate all contributions, no matter how small!

## License

By contributing to the OCR Detection Library, you agree that your contributions will be licensed under the MIT License.