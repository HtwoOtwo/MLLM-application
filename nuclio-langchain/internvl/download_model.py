from huggingface_hub import snapshot_download

REPO_ID = "OpenGVLab/InternVL2-1B"

snapshot_download(repo_id=REPO_ID, endpoint="https://hf-mirror.com", local_dir="OpenGVLab/InternVL2-1B")