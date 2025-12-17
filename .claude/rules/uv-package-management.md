---
paths:
  - pyproject.toml
  - "*.py"
---

# UV Package Management Rules

## Absolute Rules

### ✅ CORRECT

```bash
# Add production dependency
uv add torch torchvision

# Add multiple packages
uv add pandas numpy scikit-learn

# Add development dependency
uv add --dev jupyter pytest ruff

# Run script
uv run python train.py

# Run Jupyter
uv run jupyter lab
```

### ❌ FORBIDDEN

```bash
pip install torch              # NEVER use pip
uv pip install torch           # NEVER use uv pip
python -m pip install torch    # NEVER use pip
```

## GPU PyTorch Installation

```bash
# Kaggle-compatible CUDA 12.1
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU version (testing only)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Common Operations

```bash
# Sync dependencies from pyproject.toml
uv sync

# Update a package
uv add torch --upgrade

# Remove a package
uv remove package-name

# Show installed packages
uv pip list

# Python version management
uv python install 3.10
uv python pin 3.10
```

## Why UV?

- **10-100x faster** than pip
- **Reliable** dependency resolution
- **Compatible** with pip, pip-tools, poetry
- **Project-centric** with pyproject.toml

## Critical Reminder

If you see any `pip install` command suggestion, immediately replace it with `uv add`.
