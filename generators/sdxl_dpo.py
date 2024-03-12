import diffusers
import inspect
import torch
from PIL import Image

from .base import BaseGenerator


class SDXLDPO(BaseGenerator):
    def __init__(self):
        super(SDXLDPO, self).__init__()

        # load pipeline
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16",
                                                         use_safetensors=True).to("cuda")

        # load finetuned model
        unet_id = "mhdang/dpo-sdxl-text2image-v1"
        unet = diffusers.UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
        pipe.unet = unet
        pipe = pipe.to("cuda")
        self.pipe = pipe

        self.num_inference_steps = (
            inspect.signature(self.pipe.__call__)
            .parameters["num_inference_steps"]
            .default
        )

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(prompt, guidance_scale=5).images[0]

        return image
