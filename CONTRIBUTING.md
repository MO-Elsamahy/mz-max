# Contributing to MZ Max Professional

Thank you for your interest in contributing to MZ Max Professional! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code Style](#code-style)
- [Testing](#testing)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and professional in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Test your changes thoroughly
6. Submit a pull request

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mz-max.git
cd mz-max

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing functionality
- **Feature additions**: Add new features or capabilities
- **Documentation**: Improve or add documentation
- **Performance improvements**: Optimize existing code
- **UI/UX improvements**: Enhance user interfaces
- **Tests**: Add or improve test coverage

### Branch Naming Convention

Use descriptive branch names with prefixes:
- `feature/`: New features (`feature/automl-optimization`)
- `bugfix/`: Bug fixes (`bugfix/memory-leak-fix`)
- `docs/`: Documentation updates (`docs/api-reference`)
- `refactor/`: Code refactoring (`refactor/data-pipeline`)

## Pull Request Process

1. **Update Documentation**: Ensure any new features are documented
2. **Add Tests**: Include tests for new functionality
3. **Update Changelog**: Add entry to CHANGELOG.md
4. **Code Quality**: Ensure code passes all quality checks
5. **Review Process**: Address feedback from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

## Issue Reporting

When reporting issues, please include:

1. **Clear Title**: Descriptive summary of the issue
2. **Environment**: Python version, OS, package version
3. **Reproduction Steps**: Detailed steps to reproduce
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Code Sample**: Minimal code that demonstrates the issue

### Issue Templates

**Bug Report:**
```markdown
**Environment:**
- Python version:
- MZ Max version:
- Operating System:

**Description:**
Clear description of the bug

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Code Sample:**
```python
# Minimal code sample
```

## Code Style

### Python Style Guidelines

We follow PEP 8 with some modifications:

- **Line Length**: Maximum 127 characters
- **Import Organization**: Use isort for import sorting
- **Code Formatting**: Use Black for code formatting
- **Type Hints**: Use type hints for all public APIs
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

Run these tools before submitting:

```bash
# Code formatting
black mz_max/

# Import sorting
isort mz_max/

# Linting
flake8 mz_max/

# Type checking
mypy mz_max/

# Run all quality checks
make quality  # or equivalent script
```

### Documentation Style

- Use clear, concise language
- Include code examples for complex features
- Update docstrings for any modified functions/classes
- Ensure examples are tested and work correctly

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── ui/            # UI component tests
└── fixtures/      # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mz_max --cov-report=html

# Run specific test file
pytest tests/test_automl.py

# Run tests with specific marker
pytest -m "not slow"
```

### Writing Tests

- Write tests for all new functionality
- Maintain high test coverage (aim for >90%)
- Use descriptive test names
- Include edge cases and error conditions
- Mock external dependencies

Example test structure:
```python
import pytest
from mz_max import AutoMLPipeline

class TestAutoMLPipeline:
    def test_initialization_with_defaults(self):
        """Test AutoMLPipeline initializes with default parameters."""
        pipeline = AutoMLPipeline()
        assert pipeline.task_type == 'classification'
        
    def test_fit_with_valid_data(self):
        """Test pipeline training with valid dataset."""
        # Test implementation
        pass
        
    def test_fit_with_invalid_data_raises_error(self):
        """Test pipeline raises appropriate error with invalid data."""
        with pytest.raises(ValueError):
            # Test implementation
            pass
```

## Performance Guidelines

- Profile code changes for performance impact
- Use appropriate data structures and algorithms
- Consider memory usage for large datasets
- Document any performance trade-offs

## Security Considerations

- Never commit sensitive information (API keys, passwords)
- Validate all user inputs
- Use secure coding practices
- Report security issues privately to maintainers

## Documentation Requirements

### API Documentation

- All public functions and classes must have docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions that may be raised

### User Documentation

- Update README.md for significant changes
- Add examples for new features
- Update installation instructions if needed
- Keep changelog up to date

## Release Process

Releases are handled by maintainers following semantic versioning:

- **Major (X.0.0)**: Breaking changes
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

## Getting Help

- **GitHub Discussions**: General questions and community support
- **GitHub Issues**: Bug reports and feature requests
- **Email**: Direct contact for sensitive issues

## Recognition

Contributors are recognized in:
- CHANGELOG.md for significant contributions
- GitHub contributors page
- Special mentions in release notes

Thank you for contributing to MZ Max Professional!
