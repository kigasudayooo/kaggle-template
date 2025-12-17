# Training Workflow

Complete pre-training checklist and workflow for Kaggle competitions.

## Pre-training Checks

### 1. GPU Verification

```bash
!nvidia-smi
```

Verify:
- GPU name and VRAM capacity
- Current VRAM usage
- GPU utilization (%)

### 2. TODO.md Update

**MANDATORY**: Update @todo.md BEFORE starting training:

```markdown
## 現在進行中

### モデル訓練 - [実験名] - [開始時刻]
- **状態**: 進行中
- **目的**: [この実験の目的]
- **モデル**: [モデル名]
- **ハイパーパラメータ**:
  - batch_size: XX
  - learning_rate: X.XXXX
  - epochs: XXX
- **期待結果**: [期待するメトリクス]
- **関連ファイル**: `src/scripts/train_expXX.py`
```

### 3. Trackio Configuration

Ensure trackio is properly configured:

```python
import trackio

trackio.init(
    project="your-competition-name",
    config={
        "model": "convnext_small",
        "img_size": 224,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
)
```

### 4. Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

## Training Command

```bash
# Basic training
uv run python src/scripts/train.py

# With arguments (use $ARGUMENTS)
uv run python src/scripts/train.py $ARGUMENTS

# With config file
uv run python src/scripts/train.py --config configs/exp01_baseline.yaml

# With Task runner (if configured)
task train -- --config configs/exp01_baseline.yaml
```

## Monitor Training

### Launch Trackio Dashboard

```bash
!trackio show --project "your-competition-name"
```

Browser will open at: http://127.0.0.1:7860

### Monitor GPU

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Recommended Hyperparameters by GPU

| GPU | VRAM | Image Size | batch_size | num_workers |
|-----|------|------------|------------|-------------|
| RTX 4090 | 24GB | 512×512 | 16-24 | 8-12 |
| RTX 4090 | 24GB | 224×224 | 32-48 | 8-12 |
| RTX 3090 | 24GB | 512×512 | 12-16 | 6-8 |
| RTX 3080 | 10GB | 512×512 | 6-8 | 4-6 |
| T4 (Kaggle) | 16GB | 224×224 | 16-32 | 2-4 |

## Training Loop Best Practices

### Use Mixed Precision Training

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

### Log Metrics with Trackio

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)

    trackio.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,
        }, f'models/best_model_fold{fold}.pth')
```

## Post-training Actions

### 1. Update TODO.md

Mark training as completed and record results:

```markdown
## 最近完了したタスク

### [完了日時] - モデル訓練 - [実験名]
- **結果**: Val Acc = X.XXX, Val Loss = X.XXX
- **学び**: [この実験から得た教訓]
- **次のアクション**: [次に試すこと]
```

### 2. Save Experiment Artifacts

```bash
# Create experiment directory
mkdir -p experiments/expXX

# Save predictions, configs, logs
cp models/best_model_fold*.pth experiments/expXX/
cp configs/expXX_config.yaml experiments/expXX/
cp logs/training.log experiments/expXX/
```

### 3. Commit Results

```bash
git add experiments/expXX/ todo.md
git commit -m "exp: Record exp

XX results (Val Acc=X.XXX)

🤖 Generated with Claude Code"
```

## Common Issues

### CUDA OOM

```
❌ RuntimeError: CUDA out of memory
✅ Solutions:
   - Reduce batch_size
   - Use gradient accumulation
   - Reduce image size
   - Use mixed precision training
```

### Slow Training

```
❌ Training taking too long
✅ Solutions:
   - Increase num_workers (but not > CPU cores)
   - Enable pin_memory=True
   - Use mixed precision training
   - Check data loading bottleneck
```

Good luck with training! 🚀
