# Code Quality Tools Guide

## Ruff - Linter & Formatter

Ruff is a Rust-powered Python linter and formatter that's 10-100x faster than traditional tools.

### Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "*.ipynb",
]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # Pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
]
ignore = [
    "E501",   # line too long (let formatter handle it)
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert usage
"notebooks/*" = ["T20", "E402"]  # Allow print, relaxed import order
```

### Usage

```bash
# Check
uv run ruff check src/

# Fix automatically
uv run ruff check --fix src/

# Format
uv run ruff format src/
```

## ty - Type Checker

ty is a next-generation Python type checker from Astral (Ruff/uv creators).

### Configuration

```toml
# pyproject.toml
[tool.ty]
python-version = "3.11"
```

### Usage

```bash
# Type check
uvx ty check

# Specific directory
uvx ty check src/
```

## pre-commit - Git Hooks

Automatically run checks before commits.

### Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: detect-private-key
      - id: check-yaml
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.3
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### Usage

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files

# Update hooks
uv run pre-commit autoupdate
```

## Best Practices

1. **Start simple**: Begin with basic rules (E, F, I) and add gradually
2. **Auto-fix**: Use `--fix` for safe automatic corrections
3. **CI integration**: Run same checks in CI/CD pipeline
4. **Per-file ignores**: Different rules for tests and notebooks
