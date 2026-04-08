#!/usr/bin/env python3
"""Upload checkpoint files to a Hugging Face dataset repository.

Examples:
  # Upload one checkpoint file to datasets/qsun2001/omg:v1/my_run/model.pt
  python tools/upload_ckpt_to_hf.py \
    --source output/base-120m/checkpoints/model.pt \
    --repo-id qsun2001/omg \
    --path-in-repo v1/base-120m/checkpoints/model.pt

  # Upload a checkpoint directory to datasets/qsun2001/omg:v1/base-120m/checkpoints/
  python tools/upload_ckpt_to_hf.py \
    --source output/base-120m/checkpoints \
    --repo-id qsun2001/omg \
    --path-in-repo v1/base-120m/checkpoints
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload checkpoint file(s) to a Hugging Face DATASET repo. "
            "By default this script targets qsun2001/omg and stores files under v1/."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local source file or directory to upload.",
    )
    parser.add_argument(
        "--repo-id",
        default="qsun2001/omg",
        help="Hugging Face dataset repo id (default: qsun2001/omg).",
    )
    parser.add_argument(
        "--path-in-repo",
        default="v1",
        help="Target path in dataset repo (default: v1).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload checkpoint(s)",
        help="Commit message on Hugging Face Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face token. If omitted, script uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist.",
    )
    return parser.parse_args()


def resolve_token(token_arg: str | None) -> str | None:
    if token_arg:
        return token_arg
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def normalize_repo_path(path_in_repo: str) -> str:
    return path_in_repo.strip().lstrip("/")


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    token = resolve_token(args.token)
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or pass --token."
        )

    path_in_repo = normalize_repo_path(args.path_in_repo)
    if not path_in_repo:
        raise ValueError("--path-in-repo cannot be empty.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    if source.is_file():
        # If user gives a folder-like path, append the filename automatically.
        target_path = path_in_repo
        if target_path.endswith("/") or "." not in Path(target_path).name:
            target_path = f"{target_path.rstrip('/')}/{source.name}" if target_path else source.name

        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=target_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            commit_message=args.commit_message,
            token=token,
        )
        print(f"Uploaded file: {source} -> datasets/{args.repo_id}/{target_path}")
        return

    api.upload_folder(
        folder_path=str(source),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message=args.commit_message,
        token=token,
    )
    print(f"Uploaded folder: {source} -> datasets/{args.repo_id}/{path_in_repo}")


if __name__ == "__main__":
    main()
