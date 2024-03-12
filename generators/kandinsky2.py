import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Kandinsky2(BaseGenerator):
    def __init__(self):
        super(Kandinsky2, self).__init__()

        self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to("cuda")
        self.neg_prompt = "low quality, bad quality"

        self.num_inference_steps = (
            inspect.signature(self.pipe.__call__)
            .parameters["num_inference_steps"]
            .default
        )

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            prior_guidance_scale=1.0,
            height=768,
            width=768,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        return image
