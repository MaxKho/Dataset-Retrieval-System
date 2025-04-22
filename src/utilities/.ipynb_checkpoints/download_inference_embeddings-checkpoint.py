# src/utilities/download_inference_embeddings.py

import os
import sys
import zipfile
import subprocess

# YOUR GDrive link + where we expect the unzipped output
GDRIVE_LINK = "https://drive.google.com/file/d/17D8X6yYMiTXzKP3oMam30ZuwPDZY6VZ6"
TARGET_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..",           # → project root
    "data", "data", "3_final_data"
))

# List out the files we consider “the embeddings bundle”
_EXPECTED = ["doc_embeddings_partial.pkl",
             "doc_embeddings_joint.pkl",
             "final_test.json"]

def download_and_unzip_from_gdrive(share_link: str = GDRIVE_LINK,
                                   dest_folder: str = TARGET_DIR,
                                   zip_name: str = "downloaded.zip"):
    # If all expected files already on disk, do nothing
    if all(os.path.exists(os.path.join(TARGET_DIR, fn)) for fn in _EXPECTED):
        print("Embeddings already present, skipping download.")
        return

    # 1) Ensure gdown is installed
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    # 2) Make the folder
    os.makedirs(dest_folder, exist_ok=True)

    # 3) Extract file ID from share link
    try:
        file_id = share_link.split('/d/')[1].split('/')[0]
    except:
        raise ValueError(f"Could not parse file ID from link: {share_link}")
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # 4) Download
    zip_path = os.path.join(dest_folder, zip_name)
    print(f"Downloading {download_url} -> {zip_path} ...")
    gdown.download(download_url, zip_path, quiet=False)

    # 5) Unzip
    print(f"Unzipping {zip_path} -> {dest_folder} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_folder)

    # 6) Clean up
    os.remove(zip_path)
    print("Download + unzip complete.")