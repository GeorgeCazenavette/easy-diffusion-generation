from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseGenerator(ABC):
    def __init__(self):
        self.extra_kwargs = {}
        self.num_inference_steps = None

    @abstractmethod
    def generate(self, prompt: str, generator: torch.Generator) -> Image:
        pass

    def __call__(self, prompt: str, generator: torch.Generator) -> Image:
        return self.generate(prompt, generator)
