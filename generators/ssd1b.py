import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class SSD1B(BaseGenerator):
    def __init__(self):
        super(SSD1B, self).__init__()

        self.pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        self.neg_prompt = "ugly, blurry, poor quality"

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
