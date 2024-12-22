import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path
from typing import Optional
import gc
from FlowEdit_utils import FlowEditSD3, FlowEditFLUX
from download_weights import download_models

class Predictor(BasePredictor):
    def setup(self):
        """Load the model configurations but not the weights"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_model_type = None
        
        print("----- Downloading models ------")
        download_models()
        print("----- END DOWNLOAD IN CACHE ------")
        
        # Store paths to model weights
        self.model_paths = {
            "FLUX": "weights/flux"
        }

    def load_model(self, model_type: str):
        """
        Load specified model into memory, clearing other models if necessary
        """
        if self.current_model_type == model_type:
            return self.current_model
            
        # Clear current model if it exists
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load requested model
        if model_type == "SD3":
            self.current_model = StableDiffusion3Pipeline.from_pretrained(
                self.model_paths["SD3"],
                torch_dtype=torch.float16,
                local_files_only=True  # Force loading from local files
            ).to(self.device)
        else:
            self.current_model = FluxPipeline.from_pretrained(
                self.model_paths["FLUX"],
                torch_dtype=torch.float16,
                local_files_only=True  # Force loading from local files
            ).to(self.device)
            
        self.current_model_type = model_type
        return self.current_model

    def predict(
        self,
        image: Path = Input(description="Input image to edit"),
        model_type: str = Input(
            description="Model to use for editing",
            choices=["FLUX"],
            default="FLUX"
        ),
        source_prompt: str = Input(description="Description of the input image"),
        target_prompt: str = Input(description="Description of desired output image"),
        num_steps: int = Input(
            description="Total number of discretization steps",
            default=28,
            ge=1,
            le=50
        ),
        src_guidance_scale: float = Input(
            description="Source prompt CFG scale",
            default=1.5,
            ge=1.0,
            le=30.0
        ),
        tar_guidance_scale: float = Input(
            description="Target prompt CFG scale",
            default=5.5,
            ge=1.0,
            le=30.0
        ),
        n_max: int = Input(
            description="Control the strength of the edit",
            default=33,
            ge=1,
            le=50
        ),
        n_min: int = Input(
            description="Minimum step for improved style edits",
            default=0,
            ge=0
        ),
        n_avg: int = Input(
            description="Average step count (improves structure at cost of runtime)",
            default=1,
            ge=1
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=42
        )
    ) -> Path:
        """Run editing on the input image"""
        # Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Load and preprocess image
        image = Image.open(image)
        # Crop image to have dimensions divisible by 16
        image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
        
        # Load appropriate model
        pipe = self.load_model(model_type)
            
        # Preprocess image
        image_src = pipe.image_processor.preprocess(image)
        image_src = image_src.to(self.device).half()

        # Encode image
        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
        x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        x0_src = x0_src.to(self.device)

        try:
            # Apply FlowEdit
            if model_type == "SD3":
                x0_tar = FlowEditSD3(
                    pipe,
                    pipe.scheduler,
                    x0_src,
                    source_prompt,
                    target_prompt,
                    "",  # negative prompt
                    num_steps,
                    n_avg,
                    src_guidance_scale,
                    tar_guidance_scale,
                    n_min,
                    n_max
                )
            else:
                x0_tar = FlowEditFLUX(
                    pipe,
                    pipe.scheduler,
                    x0_src,
                    source_prompt,
                    target_prompt,
                    "",  # negative prompt
                    num_steps,
                    n_avg,
                    src_guidance_scale,
                    tar_guidance_scale,
                    n_min,
                    n_max
                )

            # Decode and postprocess
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            image_tar = pipe.image_processor.postprocess(image_tar)

            # Save and return result
            output_path = Path("/tmp/output.png")
            image_tar[0].save(output_path)
            return output_path
            
        finally:
            # Clean up CUDA cache after processing
            torch.cuda.empty_cache()
            gc.collect()