import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Openjourney(BaseGenerator):
    def __init__(self):
        super(Openjourney, self).__init__()

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "prompthero/openjourney", torch_dtype=torch.float16, safety_checker=None
        )
        self.pipe.to("cuda")

        self.num_inference_steps = 50

    def generate(self, prompt, generator) -> Image:
        prompt = "mdjrny-v4 style, " + prompt

        image = self.pipe(
            prompt, num_inference_steps=self.num_inference_steps, generator=generator
        ).images[0]

        return image
