# Testing Infrastructure

This directory contains the test suite for the Video Super-Resolution project.

## Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared pytest fixtures
├── unit/                # Unit tests
│   └── __init__.py
├── integration/         # Integration tests
│   └── __init__.py
└── test_infrastructure_validation.py  # Validation tests
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov

# Run specific test markers
poetry run pytest -m unit        # Run only unit tests
poetry run pytest -m integration # Run only integration tests
poetry run pytest -m "not slow"  # Skip slow tests

# Run tests in verbose mode
poetry run pytest -v

# Run a specific test file
poetry run pytest tests/test_infrastructure_validation.py
```

## Writing Tests

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use the fixtures defined in `conftest.py`
4. Mark tests appropriately with `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow`

## Coverage

Coverage reports are generated automatically when running tests:
- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`
- Terminal report: Shown after test run

The project is configured to require 80% code coverage.