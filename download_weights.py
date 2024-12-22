import os
import torch
from huggingface_hub import snapshot_download
from diffusers import FluxPipeline, StableDiffusion3Pipeline

def download_models():
    """
    Download model weights from HuggingFace.
    Creates a weights directory and downloads all required model files.
    """
    os.makedirs("weights", exist_ok=True)
    
    print("Downloading SD3 weights...")
    sd3_path = snapshot_download(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        local_dir="weights/sd3",
        ignore_patterns=["*.safetensors", "*.bin", "*.onnx"],
    )
    
    print("Downloading FLUX weights...")
    flux_path = snapshot_download(
        "black-forest-labs/FLUX.1-dev",
        local_dir="weights/flux",
        ignore_patterns=["*.safetensors", "*.bin", "*.onnx"],
    )
    
    return {"sd3_path": sd3_path, "flux_path": flux_path}
