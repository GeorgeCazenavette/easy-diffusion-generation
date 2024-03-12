import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Wurstchen2(BaseGenerator):
    def __init__(self):
        super(Wurstchen2, self).__init__()

        self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
            "warp-diffusion/wuerstchen",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        self.num_inference_steps = (
            inspect.signature(self.pipe.__call__)
            .parameters["num_inference_steps"]
            .default
        )

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            prior_guidance_scale=4.0,
            decoder_guidance_scale=0.0,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        return image
