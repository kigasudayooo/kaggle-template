# Python Project Setup (Universal)

## Prerequisites

Install uv:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Project Initialization

```bash
# Create project directory
mkdir my-project && cd my-project

# Initialize Git
git init

# Initialize uv project
uv init
uv python pin 3.11
uv sync
```

## Add Dependencies

```bash
# Core dependencies
uv add requests httpx pydantic

# Development tools
uv add --dev pytest ruff mypy pre-commit
```

## Setup Code Quality

```bash
# Install pre-commit hooks
uv run pre-commit install

# Create .pre-commit-config.yaml (use code-quality-tools rule)
```

## Project Structure

```
my-project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── main.py
├── tests/
│   └── test_main.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## Next Steps

1. Write code in `src/`
2. Write tests in `tests/`
3. Run: `uv run pytest`
4. Commit: `/git:commit`
