import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class Kandinsky3(BaseGenerator):
    def __init__(self):
        super(Kandinsky3, self).__init__()

        pipe = diffusers.Kandinsky3Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        )
        pipe.to("cuda")
        self.pipe = pipe
        self.num_inference_steps = 25

    def generate(self, prompt, generator) -> Image:
        image = self.pipe(
            prompt, num_inference_steps=self.num_inference_steps, generator=generator
        ).images[0]

        return image
