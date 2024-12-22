# Cog implementation of FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models

[![Zero-Shot Image Editing](https://img.shields.io/badge/zero%20shot-image%20editing-Green)]([https://github.com/topics/video-editing](https://github.com/topics/text-guided-image-editing))
[![Python](https://img.shields.io/badge/python-3.8+-blue?python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-38/)
![PyTorch](https://img.shields.io/badge/torch-2.0.0-red?PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i image="./image.png" -i model_type="FLUX" -i source_prompt="a cat sitting in the grass"  -i target_prompt="a puppy sitting in the grass"

![](./output.png)

## Recommend settings 

### FLUX

- Steps = 28
- Guidance (CFG) = 1.5
- TAR guidance = 5.5
– n_max = 24

### SD3

- Steps = 50
- Guidance (CFG) = 3.5
- TAR guidance = 13.5
– n_max = 33


## Credits: FlowEdit

[Project](https://matankleiner.github.io/flowedit/) | [Arxiv](https://arxiv.org/abs/2412.08629) | [Demo](https://huggingface.co/spaces/fallenshock/FlowEdit) | [ComfyUI](#comfyui-implementation-for-different-models)


### Official Pytorch implementation of the paper: "FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models"

![](https://github.com/fallenshock/FlowEdit/raw/main/imgs/teaser.png)


## License

The code in this repository is licensed under the [Apache-2.0 License](LICENSE).

Flux Dev falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

`FLUX.1 [dev]` fine-tuned weights and their outputs are non-commercial by default, but can be used commercially when running on Replicate.