import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class DreamlikePhoto(BaseGenerator):
    def __init__(self):
        super(DreamlikePhoto, self).__init__()

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16
        ).to("cuda")

        self.num_inference_steps = 50

    def generate(self, prompt, generator) -> Image:
        prompt = "photo, " + prompt

        image = self.pipe(
            prompt, num_inference_steps=self.num_inference_steps, generator=generator
        ).images[0]

        return image
