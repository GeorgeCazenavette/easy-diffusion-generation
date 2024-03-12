import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Amused(BaseGenerator):
    def __init__(self):
        super(Amused, self).__init__()

        pipe = diffusers.AmusedPipeline.from_pretrained(
            "amused/amused-512", variant="fp16", torch_dtype=torch.float16
        )

        pipe.transformer = torch.compile(pipe.transformer)

        pipe.vqvae.to(torch.float32)  # vqvae is producing nans n fp16
        pipe = pipe.to("cuda")

        self.pipe = pipe

        self.num_inference_steps = (
            inspect.signature(self.pipe.__call__)
            .parameters["num_inference_steps"]
            .default
        )

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        return image
