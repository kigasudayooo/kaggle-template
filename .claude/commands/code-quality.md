# Code Quality Check (Universal)

## Run All Checks

```bash
# Linting
uv run ruff check src/

# Formatting check
uv run ruff format --check src/

# Type checking
uvx ty check

# Tests
uv run pytest
```

## Fix Issues

```bash
# Auto-fix lint issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

## Pre-commit

```bash
# Run pre-commit on all files
uv run pre-commit run --all-files
```

## CI/CD Integration

Add to GitHub Actions:
```yaml
- name: Run checks
  run: |
    uv run ruff check src/
    uvx ty check
    uv run pytest
```
