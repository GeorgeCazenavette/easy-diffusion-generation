import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class SDTurbo(BaseGenerator):
    def __init__(self):
        super(SDTurbo, self).__init__()

        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to("cuda")
        self.pipe = pipe
        self.num_inference_steps = 1

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        return image
