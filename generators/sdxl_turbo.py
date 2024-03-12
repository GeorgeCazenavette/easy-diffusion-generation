import diffusers
from PIL import Image

from .base import BaseGenerator


class SDXLTurbo(BaseGenerator):
    def __init__(self):
        super(SDXLTurbo, self).__init__()

        pipe = diffusers.DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo", revision="refs/pr/4"
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
