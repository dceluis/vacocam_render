import os
import wandb

artifact_dir = "/notebooks/artifacts"

api = wandb.Api()

WANDB_NAMESPACE = "dceluis/YOLOv8"

def download_artifact(name, alias="latest", run=None):
    if run:
        artifact = run.use_artifact(f"{name}:{alias}")
    else:
        artifact = api.artifact(f"{WANDB_NAMESPACE}/{name}:{alias}")

    artifact_location = f"{artifact_dir}/{name}:{artifact.version}"

    if not os.path.exists(artifact_location):
        artifact.download(artifact_location)
    
    return (artifact, artifact_location)