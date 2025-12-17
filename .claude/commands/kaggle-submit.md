# Kaggle Submission Checklist

Complete pre-submission checklist for Kaggle competition notebooks.

## Critical Constraints

Kaggle notebooks have specific limitations. Verify ALL items below.

## Pre-submission Checklist

### 1. Data Augmentation

- [ ] **NOT using albumentations** (Kaggle環境で動作しない)
- [ ] **USING torchvision.transforms** (唯一の信頼できるオプション)

### 2. Model Loading

- [ ] **weights_only=False** specified in torch.load()
- [ ] Model checkpoint paths use `/kaggle/input/...`

### 3. Model Definition

- [ ] **Complete model class definition in notebook** (外部importは不可)
- [ ] All custom layers defined in notebook
- [ ] No external .py file imports for model architecture

### 4. Paths

- [ ] All data paths use `/kaggle/input/[competition-name]/...`
- [ ] Model checkpoint paths use `/kaggle/input/[dataset-name]/...`
- [ ] Output paths use `/kaggle/working/...`

### 5. Submission File

- [ ] `submission.csv` in correct format
- [ ] Column names match sample_submission.csv
- [ ] Number of rows matches test set size
- [ ] No missing values (unless explicitly allowed)

### 6. Notebook Configuration

- [ ] Internet: OFF (unless required)
- [ ] GPU: ON (if using GPU)
- [ ] Accelerator: GPU T4 x2 (or as needed)

## Standard Notebook Structure

Your notebook should follow this structure:

```python
# 1. Imports
import pandas as pd
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from pathlib import Path

# 2. Configuration
class CFG:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 224
    batch_size = 32
    num_workers = 2
    n_folds = 5
    model_name = 'convnext_small'
    num_classes = 10

# 3. Data Paths
DATA_DIR = Path('/kaggle/input/[competition-name]')
MODEL_DIR = Path('/kaggle/input/[your-model-dataset]')
OUTPUT_DIR = Path('/kaggle/working')

# 4. Transform (torchvision ONLY)
transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. Model Definition (COMPLETE definition in notebook)
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name,
            pretrained=False,  # Using trained weights
            num_classes=CFG.num_classes
        )

    def forward(self, x):
        return self.backbone(x)

# 6. Model Loading
model = YourModel().to(CFG.device)
checkpoint = torch.load(
    MODEL_DIR / 'best_model.pth',
    map_location=CFG.device,
    weights_only=False  # REQUIRED in PyTorch 2.6+
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 7. Inference
# [Your inference code here]

# 8. Create Submission
submission = pd.DataFrame({
    'id': test_ids,
    'target': predictions
})
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
```

## Common Issues

### albumentations Error

```
❌ ImportError: No module named 'albumentations'
✅ Solution: Use torchvision.transforms instead
```

### weights_only Error

```
❌ FutureWarning: weights_only=False deprecated
✅ Solution: Explicitly set weights_only=False
```

### Model Not Found

```
❌ ModuleNotFoundError: No module named 'your_model'
✅ Solution: Define complete model class in notebook
```

## Final Verification

Before clicking "Submit":

1. Run "Restart & Run All" - ensure notebook runs without errors
2. Check submission.csv format matches sample_submission.csv
3. Verify output file exists at `/kaggle/working/submission.csv`
4. Review notebook logs for any warnings

## Submission

1. Save Version
2. Click "Submit to Competition"
3. Wait for scoring
4. Record score in @todo.md with experiment details

Good luck! 🎯
