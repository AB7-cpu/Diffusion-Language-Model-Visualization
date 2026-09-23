import os
import sys
from pathlib import Path

# Redirecting Hugging Face cache to project directory.
LOCAL_HF_CACHE = Path(__file__).parent / ".hf_cache"
LOCAL_HF_CACHE.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE.resolve())
os.environ["TRANSFORMERS_CACHE"] = str(LOCAL_HF_CACHE.resolve())

from huggingface_hub import snapshot_download

def main():
    repo_id = "GSAI-ML/LLaDA-8B-Instruct"
    target_dir = Path(__file__).parent / "checkpoints" / "LLaDA-8B-Instruct"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(f"Target repository: {repo_id}")
    print(f"Destination folder: {target_dir.resolve()}")
    print(f"HF cache folder:    {LOCAL_HF_CACHE.resolve()}")
    print("==================================================")
    print("Starting download...")
    sys.stdout.flush()

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir.resolve()),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )

    print(f"\nSuccessfully downloaded {repo_id} to {target_dir.resolve()}!")

if __name__ == "__main__":
    main()
