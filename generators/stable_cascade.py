import inspect

import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class StableCascade(BaseGenerator):
    def __init__(self):
        super(StableCascade, self).__init__()

        device = "cuda"

        self.prior = diffusers.StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                           torch_dtype=torch.bfloat16).to(device)
        self.decoder = diffusers.StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",
                                                               torch_dtype=torch.float16).to(device)

    def generate(self, prompt, generator) -> Image:
        prior_output = self.prior(
            prompt=prompt,
            height=1024,
            width=1024,
            negative_prompt="",
            guidance_scale=4.0,
            num_images_per_prompt=1,
            num_inference_steps=20
        )
        image = self.decoder(
            image_embeddings=prior_output.image_embeddings.half(),
            prompt=prompt,
            negative_prompt="",
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=10
        ).images[0]

        return image
