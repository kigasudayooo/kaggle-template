---
paths:
  - notebooks/**/*.ipynb
  - notebooks/**/*.py
---

# Kaggle Submission Constraints

## Absolute Requirements

1. **DO NOT use albumentations** - Kaggle environment does not support it
2. **USE torchvision.transforms** - Only reliable option
3. **weights_only=False** - Required for torch.load() in PyTorch 2.6+
4. **Complete model definition in notebook** - No external imports

## Standard Patterns

### Transform (torchvision only)

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Model Loading

```python
import torch

checkpoint = torch.load(
    path,
    map_location=device,
    weights_only=False  # Required in PyTorch 2.6+
)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Complete Notebook Structure

```python
# 1. Imports
import pandas as pd
import torch
import timm
from torchvision import transforms
from PIL import Image

# 2. Model Definition (complete class definition in notebook)
class YourModel(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

# 3. Configuration
class CFG:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 224
    n_folds = 5

# 4. Transform
transform = transforms.Compose([...])

# 5. Model Loading
checkpoint = torch.load(path, map_location=CFG.device, weights_only=False)

# 6. Inference
# 7. Create Submission
```

## Pre-submission Checklist

- [ ] albumentationsを使用していないか
- [ ] torchvision transformsを使用しているか
- [ ] モデルクラスがノートブック内で完全に定義されているか
- [ ] weights_only=Falseを指定しているか
- [ ] パスが/kaggle/input/...になっているか
- [ ] submission.csvが正しいフォーマットか
