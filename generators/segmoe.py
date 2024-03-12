import inspect

import diffusers, segmoe
import torch
from PIL import Image

from .base import BaseGenerator


class SegMoE(BaseGenerator):
    def __init__(self):
        super(SegMoE, self).__init__()

        pipe = segmoe.SegMoEPipeline("segmind/SegMoE-4x2-v0", device = "cuda", torch_dtype=torch.float16)
        # pipe.to("cuda")

        self.pipe = pipe

        self.neg_prompt = "nsfw, bad quality, worse quality"

        # self.num_inference_steps = (
        #     inspect.signature(self.pipe.__call__)
        #     .parameters["num_inference_steps"]
        #     .default
        # )

        self.num_inference_steps = 25

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            guidance_scale=7.5,
            height=1024,
            width=1024,
        ).images[0]

        return image
