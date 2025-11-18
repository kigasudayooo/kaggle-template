# Kaggle Competition Project

## Project Overview
Machine learning competition project with focus on model development, experiment tracking, and reproducible results.

Competition: [Competition name and link]
Evaluation Metric: [Metric name - e.g., RMSE, F1, AUC]
Current Best Score: [CV score / LB score]

## Tech Stack & Versions
- Python: 3.10+ (match Kaggle environment)
- PyTorch: 2.1.0+ with CUDA 12.1 support
- Core Libraries: pandas, numpy, scikit-learn, timm, Pillow
- Experiment Tracking: trackio (wandb-compatible API)
- Package Manager: uv (NOT pip - use `uv add` for all dependencies)
- Environment: pyproject.toml + uv.lock for reproducibility

## Project Structure
```
project/
├── data/                    # Data files (gitignored)
│   ├── raw/                # Original competition data
│   ├── processed/          # Preprocessed datasets
│   └── sample/             # Small samples for testing (git tracked)
├── notebooks/              # Jupyter notebooks
│   ├── eda/               # Exploratory data analysis
│   ├── experiments/       # Experiment notebooks
│   └── submission/        # Kaggle submission notebooks
├── src/
│   ├── data/              # Dataset classes, data loaders
│   ├── models/            # Model architectures
│   ├── features/          # Feature engineering
│   ├── utils/             # Utility functions
│   └── scripts/           # Training/inference scripts
├── configs/               # YAML configuration files
├── models/                # Trained models (gitignored)
├── experiments/           # Experiment outputs (gitignored)
├── submissions/           # Submission files (gitignored)
├── docs/                  # Documentation
│   ├── 01_getting_started/
│   ├── 02_experiments/
│   ├── 03_architecture/
│   └── 04_kaggle_submission/
├── todo.md                # TODO management (CRITICAL - update before any work)
├── CLAUDE.md              # This file
└── pyproject.toml         # uv project configuration
```

## Environment Setup

### CRITICAL: Use uv for ALL package management
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
uv python install 3.10
uv python pin 3.10
uv sync

# Add packages (NEVER use pip install)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv add pandas numpy scikit-learn pillow timm tqdm pyyaml trackio

# Development tools
uv add --dev jupyter pytest black mypy

# Run scripts with uv
uv run python train.py
uv run jupyter lab
```

## Data Configuration
- **Local**: `DATA_PATH=./data/raw/`
- **Kaggle**: `DATA_PATH=/kaggle/input/[competition-name]/`
- Use environment variable `DATA_PATH` in code, configure via `.env` file

Data Schema:
- `train.csv`: [column names and types]
- `test.csv`: [column names and types]
- Target: [target variable name and description]

## Common Commands

### Training & Evaluation
```bash
# Train model
uv run python src/scripts/train.py --config configs/baseline.yaml

# Quick test run (small epochs)
uv run python src/scripts/train.py --config configs/baseline.yaml training.epochs=2

# Cross-validation
uv run python src/scripts/train_cv.py --config configs/baseline.yaml

# Inference
uv run python src/scripts/predict.py --model-path models/best_model.pth
```

### Experiment Tracking
```bash
# Launch trackio dashboard (ALWAYS check after training starts)
trackio show --project "project-name"
# Opens browser at http://127.0.0.1:7860
```

### Development
```bash
# Jupyter Lab
uv run jupyter lab

# Linting
uv run black src/ && uv run mypy src/

# Testing
uv run pytest tests/ -v
```

### Kaggle Data
```bash
# Download competition data
uv run kaggle competitions download -c competition-name
uv run kaggle datasets download -d username/dataset-name
```

## Experiment Tracking with trackio

### MANDATORY: Always use trackio for training
```python
import trackio

# Initialize (at start of training script)
trackio.init(
    project="project-name",
    config={
        "model": "convnext_small",
        "img_size": 224,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
        # Include ALL hyperparameters
    }
)

# Log metrics (in training loop)
trackio.log({
    "epoch": epoch,
    "train/loss": train_loss,
    "train/metric": train_metric,
    "val/loss": val_loss,
    "val/metric": val_metric,
    "learning_rate": optimizer.param_groups[0]['lr'],
})

# Finish (at end of training)
trackio.finish()
```

### After starting training
IMMEDIATELY run: `trackio show --project "project-name"` to monitor in real-time

## TODO Management (CRITICAL)

### ABSOLUTE RULE: Update todo.md BEFORE any work
```
Wrong workflow:
1. Start work
2. Do work
3. Update todo.md (maybe)

Correct workflow:
1. Open todo.md
2. Add/update task description
3. Commit: "docs: Add [task] to TODO"
4. Start work
5. Mark complete immediately after finishing
6. Commit: "docs: Mark [task] as completed"
```

### When to update todo.md (mandatory triggers)
- Before creating/editing any file
- Before starting any experiment/training
- When encountering bugs/errors
- Before data processing/analysis
- Before creating inference notebooks
- Before refactoring/cleanup

## Kaggle Submission Notebooks

### CRITICAL Constraints
1. **NEVER use albumentations** - not available in Kaggle environment
2. **ALWAYS use torchvision.transforms** - only reliable option
3. **Model definitions MUST be complete in notebook** - no external imports
4. **Use weights_only=False** for torch.load() in PyTorch 2.6+

### Standard Structure
```python
# 1. Imports (pandas, torch, timm, torchvision, PIL)
# 2. Model Definition (complete class definitions)
# 3. Configuration (paths, device, transforms)
# 4. Data Loading (test.csv)
# 5. Model Loading (K-fold ensemble)
# 6. Inference Function
# 7. Generate Predictions
# 8. Create Submission (submission.csv)
```

### Example Transform (torchvision only)
```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Image loading
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image)
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints for all functions: `def func(x: int) -> str:`
- Docstrings for all public functions (Google style)
- Variable names: descriptive, minimum 3 characters
- NEVER use single-letter variables except i, j, k for loops

### ML/DL Specific Patterns
```python
# Always set seeds for reproducibility
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Use no_grad for inference
with torch.no_grad():
    predictions = model(inputs)

# DataLoader settings
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # For GPU training
)
```

### File Organization
- One model per file in `src/models/`
- Dataset classes in `src/data/`
- Utility functions grouped by purpose in `src/utils/`
- Training scripts in `src/scripts/`
- Config files named descriptively: `configs/convnext_224px_5fold.yaml`

## Configuration Management

### YAML Structure
```yaml
# configs/baseline.yaml
model:
  name: "convnext_small"
  pretrained: true
  num_classes: 1

data:
  img_size: 224
  batch_size: 32
  num_workers: 4

training:
  epochs: 100
  lr: 0.0001
  optimizer: "AdamW"
  weight_decay: 0.01
  scheduler: "CosineAnnealingLR"

logging:
  use_trackio: true
  trackio:
    project: "project-name"
```

## Important Constraints & Warnings

### CRITICAL Rules
- **NEVER commit data files** - use .gitignore
- **ALWAYS use torch.no_grad()** for inference
- **NEVER modify files in data/raw/** - raw data is immutable
- **ALWAYS validate before submission** - check shapes, dtypes
- **NEVER use albumentations in Kaggle notebooks**
- **ALWAYS use uv add, NEVER pip install**

### Known Issues
- Kaggle Python environment: 3.10.13 (as of 2025-01)
- PyTorch in Kaggle: 2.1.2 with CUDA 12.1
- Memory constraints: Kaggle GPU has 16GB VRAM
- Time limit: 9 hours for GPU notebooks

## Git Workflow

### Branch Naming
- `feature/description` - new features
- `experiment/exp-name` - experiments
- `fix/bug-description` - bug fixes
- `docs/update-description` - documentation

### Commit Messages (Conventional Commits)
```bash
feat: Add ConvNeXt training script
fix: Resolve checkpoint loading error
exp: Record 5-fold CV results (R²=0.58)
docs: Update TODO with experiment status
chore: Add dependency to pyproject.toml
```

### Pre-commit Checklist
- [ ] todo.md is updated
- [ ] No large data files included
- [ ] Code passes linting: `uv run black src/`
- [ ] Tests pass: `uv run pytest`
- [ ] No sensitive data in commits

## Cross-Validation Strategy

### CRITICAL: Use appropriate CV strategy
```python
from sklearn.model_selection import StratifiedKFold, GroupKFold

# For classification (stratify by target)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For time-series or grouped data (prevent data leakage)
gkf = GroupKFold(n_splits=5)
# group by date, user_id, or other grouping variable
```

NEVER use simple KFold for imbalanced data or grouped/temporal data

## Model Design Patterns

### Standard Model Wrapper
```python
import torch
import torch.nn as nn
import timm

class ImageModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
```

### Checkpoint Saving
```python
# Save with metadata
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config,
}, f'models/checkpoint_fold{fold}_epoch{epoch}.pth')

# Load with proper handling
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False  # Required for PyTorch 2.6+
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Error Handling & Logging

### Standard Error Pattern
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    result = process_data(data)
    logger.info(f"Successfully processed {len(result)} samples")
except ValueError as e:
    logger.error(f"Data validation failed: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
```

## Performance Optimization

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Accumulation
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Documentation Standards

### Code Comments
```python
def calculate_metric(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate competition metric.
    
    Args:
        predictions: Model predictions, shape (N,)
        targets: Ground truth labels, shape (N,)
    
    Returns:
        Metric score (higher is better)
    
    Note:
        Uses sklearn implementation for consistency with Kaggle scoring
    """
    pass
```

### Experiment Documentation
After each significant experiment, update `docs/02_experiments/`:
- Hypothesis and motivation
- Model architecture and hyperparameters
- CV score and LB score (if submitted)
- What worked / what didn't
- Next steps

## Security & Best Practices

### DO:
- Use environment variables for API keys: `os.getenv('KAGGLE_KEY')`
- Store credentials in `.env` (gitignored)
- Validate all input data shapes and types
- Use configuration files for hyperparameters
- Log all experiments with trackio
- Comment complex algorithms
- Write tests for data processing functions

### DON'T:
- Hard-code file paths
- Commit API keys or passwords
- Skip data validation
- Train without setting random seeds
- Use deprecated PyTorch APIs
- Mix albumentations in submission notebooks
- Use pip install (use uv add)

## Debugging Tips

### Common Issues
1. **CUDA OOM**: Reduce batch_size, use gradient accumulation
2. **Slow training**: Check num_workers, pin_memory, use mixed precision
3. **Poor CV/LB correlation**: Check for data leakage, GroupKFold
4. **Kaggle submission errors**: Verify model definitions are in notebook
5. **Import errors in Kaggle**: Check library versions match

### Diagnostic Commands
```bash
# Check GPU memory
nvidia-smi

# Profile training
uv run python -m torch.utils.bottleneck train.py

# Check data pipeline
uv run python -m torch.utils.data.dataloader_benchmark

# Verify environment
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Resources & References
- Kaggle API: https://github.com/Kaggle/kaggle-api
- timm documentation: https://timm.fast.ai/
- PyTorch documentation: https://pytorch.org/docs/
- trackio repository: https://github.com/gradio-app/trackio

---
Last Updated: 2025-01-06
