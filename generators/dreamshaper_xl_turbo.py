import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class DreamshaperXLTurbo(BaseGenerator):
    def __init__(self):
        super(DreamshaperXLTurbo, self).__init__()

        pipe = diffusers.AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-xl-turbo', torch_dtype=torch.float16,
                                                         variant="fp16")
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        self.pipe = pipe

        self.num_inference_steps = 6

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(prompt, num_inference_steps=self.num_inference_steps, guidance_scale=2).images[0]


        return image
