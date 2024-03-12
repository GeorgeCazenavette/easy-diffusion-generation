import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class PixArt1024(BaseGenerator):
    def __init__(self):
        super(PixArt1024, self).__init__()

        self.pipe = diffusers.PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024x1024", torch_dtype=torch.float16
        ).to("cuda")
        # self.pipe.transformer = torch.compile(
        #     self.pipe.transformer, mode="reduce-overhead", fullgraph=True
        # )

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
