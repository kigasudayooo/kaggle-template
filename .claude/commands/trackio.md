# Trackio - Experiment Tracking

Lightweight, local-first experiment tracking for Kaggle competitions.

## What is Trackio?

**Trackio** is a Hugging Face-developed experiment tracking library:

- **Weights & Biases-compatible API**: Same interface as wandb
- **Local-first**: Data stored in local SQLite database
- **Gradio Dashboard**: Browser-based visualization
- **Free**: No cloud costs, no account required
- **Offline-friendly**: Works without internet

## Installation

```bash
uv add trackio
```

## Basic Usage

### 1. Initialize Tracking

```python
import trackio

# Initialize experiment
trackio.init(
    project="kaggle-competition-name",
    config={
        # Model configuration
        "model": "convnext_small",
        "pretrained": True,
        "img_size": 224,
        "num_classes": 10,

        # Training configuration
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "scheduler": "CosineAnnealingLR",

        # Data configuration
        "train_split": 0.8,
        "n_folds": 5,
        "augmentation": "medium",

        # Hardware
        "device": "cuda",
        "num_workers": 4,
    }
)
```

### 2. Log Metrics in Training Loop

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0

    for batch in train_loader:
        # ... training code ...
        pass

    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            # ... validation code ...
            pass

    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset)

    # Log metrics
    trackio.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
```

### 3. Log Additional Information

```python
# Log best model checkpoint
if val_acc > best_acc:
    best_acc = val_acc
    trackio.log({"best_val_acc": best_acc})

# Log confusion matrix, images, etc.
trackio.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )
})

# Log sample predictions
trackio.log({
    "sample_predictions": [
        trackio.Image(img, caption=f"Pred: {pred}, True: {true}")
        for img, pred, true in zip(images[:10], predictions[:10], labels[:10])
    ]
})
```

### 4. Finish Tracking

```python
# At the end of training
trackio.finish()
```

## View Dashboard

### Launch Trackio Dashboard

```bash
trackio show --project "kaggle-competition-name"
```

Browser will automatically open at: http://127.0.0.1:7860

### Dashboard Features

- **Runs table**: Compare all experiment runs
- **Metrics charts**: Interactive plots of loss, accuracy, etc.
- **Hyperparameter comparison**: See which configs work best
- **System metrics**: GPU utilization, memory usage
- **Artifacts**: View logged images, confusion matrices

## Cross-Validation Tracking

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Initialize fold-specific run
    trackio.init(
        project="kaggle-competition-name",
        name=f"exp01_fold{fold}",
        config={
            **base_config,
            "fold": fold,
        }
    )

    # Train model for this fold
    model = create_model()
    train_fold(model, train_idx, val_idx)

    # Log fold results
    trackio.log({
        f"fold{fold}/val_acc": val_acc,
        f"fold{fold}/val_loss": val_loss,
    })

    trackio.finish()

# Log averaged results
avg_val_acc = np.mean([...])
print(f"Average 5-fold CV Accuracy: {avg_val_acc:.4f}")
```

## Integration with Training Scripts

### Complete Training Script Example

```python
import trackio
import torch
from pathlib import Path

# Configuration
config = {
    "model": "convnext_small",
    "img_size": 224,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 100,
}

# Initialize tracking
trackio.init(project="my-competition", config=config)

# Training loop
for epoch in range(config["epochs"]):
    train_metrics = train_epoch(model, train_loader)
    val_metrics = validate(model, val_loader)

    # Log all metrics
    trackio.log({
        "epoch": epoch,
        **train_metrics,
        **val_metrics,
        "lr": optimizer.param_groups[0]['lr'],
    })

    # Save best model
    if val_metrics["val/accuracy"] > best_acc:
        best_acc = val_metrics["val/accuracy"]
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, 'models/best_model.pth')

# Finish
trackio.finish()
print(f"Training complete! Best Val Acc: {best_acc:.4f}")
```

## Comparison with Weights & Biases

| Feature | Trackio | Weights & Biases |
|---------|---------|------------------|
| **Cost** | Free | Free tier limited |
| **Data storage** | Local SQLite | Cloud |
| **Internet required** | No | Yes |
| **Setup** | `uv add trackio` | Account + API key |
| **API** | W&B-compatible | Native |
| **Dashboard** | Gradio (local) | Web (cloud) |

## Tips

1. **Always initialize trackio at the start of training**
2. **Use descriptive project names** (e.g., "titanic-survival-prediction")
3. **Log hyperparameters in config** for easy comparison
4. **Use hierarchical metric names** (e.g., "train/loss", "val/loss")
5. **Check dashboard regularly** during training
6. **Commit trackio.db to git** (it's small and valuable)

## Data Location

Trackio stores data in SQLite database:

```
.trackio/
└── trackio.db
```

This file is typically small (< 10MB) and can be committed to git.

Happy tracking! 📊
