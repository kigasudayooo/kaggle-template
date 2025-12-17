# Kaggle Project Setup

Complete setup guide for Kaggle competition projects using uv.

## 1. UV Installation

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Homebrew (macOS)

```bash
brew install uv
```

## 2. Project Initialization

```bash
# Navigate to project directory
cd $ARGUMENTS

# Initialize uv project
uv init

# Set Python version (Kaggle-compatible)
uv python install 3.10
uv python pin 3.10

# Initial sync
uv sync
```

## 3. Install Core Dependencies

```bash
# PyTorch with CUDA 12.1 (Kaggle-compatible)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Data science essentials
uv add pandas numpy scikit-learn

# Computer vision
uv add timm pillow

# Utilities
uv add tqdm pyyaml

# Experiment tracking
uv add trackio
```

## 4. Install Development Tools

```bash
# Development dependencies
uv add --dev jupyter pytest ruff pre-commit

# Install pre-commit hooks
uv run pre-commit install
```

## 5. Verify Installation

```bash
# Check GPU availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check installed packages
uv pip list
```

## 6. Initial Project Structure

Create the following directories:

```bash
mkdir -p data/{raw,processed,sample}
mkdir -p notebooks/{eda,experiments,submission}
mkdir -p src/{data,models,features,utils,scripts}
mkdir -p configs
mkdir -p models
mkdir -p experiments
mkdir -p tests
```

## 7. Initialize Git (if not already)

```bash
git init
git add .
git commit -m "chore: Initial project setup with uv

🤖 Generated with Claude Code"
```

## Next Steps

1. Review [CLAUDE.md](../../../CLAUDE.md) for project guidelines
2. Check [todo.md](../../../todo.md) and update it before starting work
3. Run `/trackio` to learn about experiment tracking
4. Start with EDA in `notebooks/eda/`

Your Kaggle project is ready! 🚀
