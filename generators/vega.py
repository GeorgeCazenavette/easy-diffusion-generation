import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Vega(BaseGenerator):
    def __init__(self):
        super(Vega, self).__init__()

        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
            "segmind/Segmind-Vega",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        pipe.to("cuda")
        self.pipe = pipe

        self.neg_prompt = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)"  # Negative prompt here

        self.num_inference_steps = (
            inspect.signature(self.pipe.__call__)
            .parameters["num_inference_steps"]
            .default
        )

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        return image
