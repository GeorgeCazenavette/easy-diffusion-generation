import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Consistency(BaseGenerator):
    def __init__(self):
        super(Consistency, self).__init__()

        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            safety_checker=None,
        ).to(torch_device="cuda", torch_dtype=torch.float32)
        self.pipe = self.pipe.to("cuda")
        self.num_inference_steps = 4

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil",
            generator=generator,
        ).images[0]

        return image
