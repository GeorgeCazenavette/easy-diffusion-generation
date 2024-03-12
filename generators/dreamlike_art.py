import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class DreamlikeArt(BaseGenerator):
    def __init__(self):
        super(DreamlikeArt, self).__init__()

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "dreamlike-art/dreamlike-diffusion-1.0", torch_dtype=torch.float16
        ).to("cuda")

        self.num_inference_steps = 50

    def generate(self, prompt, generator) -> Image:
        prompt = "dreamlikeart, " + prompt

        image = self.pipe(
            prompt, num_inference_steps=self.num_inference_steps, generator=generator
        ).images[0]

        return image
