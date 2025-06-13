import wandb
import os
from pathlib import Path
from typing import Dict, Any, Optional

def init_wandb(config: Dict[str, Any], project_name: Optional[str] = None) -> None:
    """Initialize Weights & Biases logging."""
    wandb_config = config.get('wandb', {})
    
    wandb.init(
        project=project_name or wandb_config.get('project', 'kaggle-competition'),
        entity=wandb_config.get('entity'),
        config=config,
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        reinit=True
    )

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to wandb."""
    wandb.log(metrics, step=step)

def log_model(model_path: str, name: str = "model") -> None:
    """Log model artifact to wandb."""
    if os.path.exists(model_path):
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

def log_submission(submission_path: str, score: float) -> None:
    """Log submission file and score to wandb."""
    if os.path.exists(submission_path):
        artifact = wandb.Artifact("submission", type="submission")
        artifact.add_file(submission_path)
        wandb.log_artifact(artifact)
        wandb.log({"submission_score": score})

def finish_wandb() -> None:
    """Finish wandb run."""
    wandb.finish()