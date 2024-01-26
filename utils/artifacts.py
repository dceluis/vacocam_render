import os
import wandb

WANDB_NAMESPACE = os.environ.get("WANDB_NAMESPACE")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR") or os.path.join(os.getcwd(), "artifacts")

api = wandb.Api()

def download_artifact(name, run=None):
    if not WANDB_NAMESPACE:
        raise ValueError("WANDB_NAMESPACE not set")

    if ":" in name:
        name, alias = name.split(":")
    else:
        name, alias = name, "latest"

    if run:
        artifact = run.use_artifact(f"{name}:{alias}")
    else:
        artifact = api.artifact(f"{WANDB_NAMESPACE}/{name}:{alias}")

    artifact_location = f"{ARTIFACT_DIR}/{name}:{artifact.version}"

    if not os.path.exists(artifact_location):
        artifact.download(artifact_location)
    
    return (artifact, artifact_location)