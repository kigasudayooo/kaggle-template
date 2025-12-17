---
paths:
  - pyproject.toml
  - .pre-commit-config.yaml
  - "**/*.py"
---

# Code Quality Toolchain (Universal)

## Recommended Tools

| Tool | Purpose | Speed |
|------|---------|-------|
| Ruff | Linter + Formatter | Rust-powered, 10-100x faster |
| ty | Type checker | Rust-powered, 10-20x faster |
| pre-commit | Git hooks | Automatic checks |

## pyproject.toml Setup

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B"]

[tool.ty]
python-version = "3.11"
```

## pre-commit Setup

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-added-large-files
      - id: trailing-whitespace
      - id: end-of-file-fixer

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.3
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

## Usage

```bash
# Linting
uv run ruff check src/
uv run ruff format src/

# Type checking
uvx ty check

# Pre-commit
uv run pre-commit install
uv run pre-commit run --all-files
```
