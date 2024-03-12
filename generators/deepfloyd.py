import diffusers
import torch
from PIL import Image

from .base import BaseGenerator


class DeepFloyd(BaseGenerator):
    def __init__(self):
        super(DeepFloyd, self).__init__()

        # stage 1
        stage_1 = diffusers.DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            watermarker=None,
            safety_checker=None,
        )
        stage_1.to("cuda")

        # stage 2
        stage_2 = diffusers.DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        stage_2.to("cuda")

        # stage 3
        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = diffusers.DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **safety_modules,
            torch_dtype=torch.float32
        )
        stage_3.to("cuda")

        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.stage_3 = stage_3

    def generate(self, prompt, generator) -> Image:
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)

        image = self.stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
            generator=generator,
        ).images

        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
            generator=generator,
        ).images

        image = self.stage_3(
            prompt=prompt, image=image, noise_level=100, generator=generator
        ).images[0]

        return image
