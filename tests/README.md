# Desktop LLM Server Tests

## Overview
This directory contains comprehensive test suites for the Desktop LLM server endpoints.

## Test Coverage
The test suite covers the following endpoints:
- `/v1/api/generate`: Completion generation
- `/v1/api/models`: Model listing
- `/v1/api/functions`: Function registration and execution
- `/v1/api/prompt-templates`: Prompt template management
- `/v1/api/special-tokens`: Special token management

## Running Tests

### Prerequisites
- Python 3.11+
- Install development dependencies:
  ```bash
  pip install -r requirements-dev.txt
  ```

### Execution
Run tests using pytest:
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_server_endpoints.py
```

## Test Configuration
- `conftest.py`: Provides global test configurations and fixtures
- `test_server_endpoints.py`: Contains endpoint-specific test cases

## Notes
- Ensure the server is running before executing tests
- Tests are designed to be idempotent and independent
- Some tests may create temporary files for testing purposes
