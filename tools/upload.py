import os
import argparse
from huggingface_hub import HfApi, login

def upload_to_hf(local_path, repo_id, path_in_repo, token=None, commit_message=None):
    """
    上传文件或文件夹到 Hugging Face Dataset
    """
    if token:
        login(token=token)
    
    api = HfApi()

    if not os.path.exists(local_path):
        print(f"❌ Error: Local path '{local_path}' does not exist.")
        return

    print(f"🚀 Preparing to upload '{local_path}' to '{repo_id}' (Dataset)...")
    print(f"📂 Target folder in repo: '{path_in_repo}'")

    try:
        if os.path.isfile(local_path):
            file_name = os.path.basename(local_path)
            target_path = f"{path_in_repo}/{file_name}" if path_in_repo else file_name
            
            print(f"Uploading single file...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=target_path,
                repo_id=repo_id,
                repo_type="dataset",  # 关键：指定是 dataset
                commit_message=commit_message or f"Upload file {file_name}"
            )
        
        elif os.path.isdir(local_path):
            print(f"Uploading folder structure...")
            api.upload_folder(
                folder_path=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",  # 关键：指定是 dataset
                commit_message=commit_message or f"Upload folder {os.path.basename(local_path)}"
            )

        print(f"✅ Success! Uploaded to: https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}")

    except Exception as e:
        print(f"❌ Failed to upload: {e}")

if __name__ == "__main__":
    # python tools/upload.py output/vae_rms_fixed_002_mse_scale_500m_objaverse/checkpoints/0126000.pt --dir denoiser/002_500M_objaverse
    parser = argparse.ArgumentParser(description="Upload VAE checkpoints to Hugging Face Dataset")
    
    parser.add_argument("path", type=str, help="Local path to the file (.pt) or folder containing checkpoints")
    
    parser.add_argument("--repo_id", type=str, default="qsun2001/omg", help="Target Hugging Face Repo ID")
    parser.add_argument("--dir", type=str, default="vae_ckpts", help="Directory inside the HF repo to store files")
    parser.add_argument("--msg", type=str, default=None, help="Commit message")
    parser.add_argument("--token", type=str, default=None, help="HF Write Token (optional if logged in via CLI)")

    args = parser.parse_args()

    upload_to_hf(
        local_path=args.path,
        repo_id=args.repo_id,
        path_in_repo=args.dir,
        token=args.token,
        commit_message=args.msg
    )