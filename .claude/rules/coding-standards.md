---
paths:
  - src/**/*.py
  - tests/**/*.py
---

# Python Coding Standards for Kaggle Projects

## Type Hints & Docstrings

- **All public functions**: type hints required
- **Public functions**: Google-style docstrings
- **Variable names**: descriptive, minimum 3 characters
- **No single-letter variables** except in loops (i, j, k acceptable)

## ML/DL Standard Patterns

### Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Inference Mode

```python
with torch.no_grad():
    predictions = model(inputs)
```

### DataLoader Configuration

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Use for GPU training
)
```

### Checkpoint Saving

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config,
}, f'models/checkpoint_fold{fold}_epoch{epoch}.pth')
```

## Important Notes

**DO NOT include style guidelines in this file:**
- Line length, quotes, indentation → Handled by Ruff
- Import order → Handled by Ruff
- Code formatting → Handled by Ruff

This file contains only **semantic** standards, not style rules.
