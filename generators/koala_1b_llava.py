import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Koala1bLlava(BaseGenerator):
    def __init__(self):
        super(Koala1bLlava, self).__init__()

        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-1b-llava-cap", torch_dtype=torch.float16)
        pipe.to("cuda")
        self.pipe = pipe

        self.num_inference_steps = (
            inspect.signature(self.pipe.__call__)
            .parameters["num_inference_steps"]
            .default
        )

        self.negative_prompt = "worst quality, low quality, illustration, low resolution"

    def generate(self, prompt, generator) -> Image:

        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        return image
