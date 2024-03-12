import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class SDXL(BaseGenerator):
    def __init__(self):
        super(SDXL, self).__init__()

        self.base = diffusers.DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base.to("cuda")

        self.refiner = diffusers.DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

        self.num_inference_steps = 40

        self.high_noise_frac = 0.8

    def generate(self, prompt, generator) -> Image:
        image = self.base(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
            generator=generator,
        ).images

        image = self.refiner(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]

        return image
