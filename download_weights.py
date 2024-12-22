import os, torch, subprocess, tarfile
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def download_models():
    """
    Download model weights from HuggingFace.
    Creates a weights directory and downloads all required model files.
    """
    os.makedirs("weights", exist_ok=True)
    
    print("Downloading FLUX weights...")
    # flux_path = snapshot_download(
    #     "black-forest-labs/FLUX.1-dev",
    #     local_dir="weights/flux",
    #     ignore_patterns=["*.bin", "*.onnx"],
    #     token=HF_TOKEN
    # )

    print("Downloading FLUX weights from Replicate...")
    flux_url = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
    
    # Download the tar file using pget
    subprocess.run(["pget", flux_url, "weights/flux_files.tar"], check=True)
    
    # Extract the tar file
    print("Extracting FLUX weights...")
    with tarfile.open("weights/flux_files.tar", "r") as tar:
        tar.extractall("weights/flux_temp")
    
    # Move files from the extracted directory to weights/flux
    # The exact path might need to be adjusted based on the tar file structure
    os.system("mv weights/flux_temp/* weights/flux/")
    
    # Clean up
    os.remove("weights/flux_files.tar")
    os.system("rm -rf weights/flux_temp")
    
    return {"flux_path": "weights/flux"}
