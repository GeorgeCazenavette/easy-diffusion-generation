import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class SD15(BaseGenerator):
    def __init__(self):
        super(SD15, self).__init__()

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        pipe = pipe.to("cuda")

        self.pipe = pipe

    def generate(self, prompt, generator) -> Image:
        # self.pipe.scheduler.num_inference_steps = self.num_inference_steps
        image = self.pipe(prompt, generator=generator).images[0]

        return image
