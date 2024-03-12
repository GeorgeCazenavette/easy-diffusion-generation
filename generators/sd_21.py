import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class SD21(BaseGenerator):
    def __init__(self):
        super(SD21, self).__init__()

        model_id = "stabilityai/stable-diffusion-2-1"

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = pipe.to("cuda")
        self.pipe = pipe

    def generate(self, prompt, generator) -> Image:
        # self.pipe.scheduler.num_inference_steps = self.num_inference_steps
        image = self.pipe(prompt, generator=generator).images[0]

        return image
