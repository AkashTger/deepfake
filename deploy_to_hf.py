import os
from huggingface_hub import HfApi, login

print("==============================================")
print("  Deploying Code to Hugging Face Spaces")
print("==============================================\n")

import sys

# Prompt user for their Hugging Face Token or take from argument
if len(sys.argv) > 1:
    token = sys.argv[1]
else:
    print("You need a Hugging Face Access Token with 'WRITE' permissions.")
    print("Get it here: https://huggingface.co/settings/tokens")
    token = input("Paste your HF Token: ").strip()

# Login securely
login(token=token)
api = HfApi()

repo_id = "DiddiAkash/deepfake-application"

print("\nUploading files. This won't overwrite your ensemble_best.safetensors model...")

# Upload folders and files individually to avoid uploading unnecessary hidden folders
folders_to_upload = ["backend", "frontend", "model_files"]
files_to_upload = ["Dockerfile", "README.md", "start.bat", "create_notebook.py"]

for folder in folders_to_upload:
    if os.path.exists(folder):
        print(f"Uploading {folder}...")
        try:
            api.upload_folder(
                folder_path=folder,
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="space",
                token=token
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to upload {folder}.")
            print(f"Detail: {e}")
            print("\nDid you definitely create a 'WRITE' token? By default tokens are 'READ' only.")
            print("Also make sure your space is exactly named: DiddiAkash/deepfake-application")
            exit(1)

for file in files_to_upload:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="space",
                token=token
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to upload {file}.")
            print(f"Detail: {e}")
            exit(1)

print("\n✅ Success! Your code has been uploaded to Hugging Face.")
print("Go to https://huggingface.co/spaces/DiddiAkash/deepfake-application to see it build!")

