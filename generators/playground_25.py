import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Playground25(BaseGenerator):
    def __init__(self):
        super(Playground25, self).__init__()

        pipe = diffusers.DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.to("cuda")
        self.pipe = pipe

        self.num_inference_steps = 50

        self.guidance_scale = 3

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        return image
