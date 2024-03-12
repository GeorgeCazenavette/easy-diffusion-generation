from .amused import Amused
from .base import BaseGenerator
from .consistency import Consistency
from .deepfloyd import DeepFloyd
from .dreamlike_art import DreamlikeArt
from .dreamlike_photo import DreamlikePhoto
from .dreamshaper_xl_turbo import DreamshaperXLTurbo
from .kandinsky2 import Kandinsky2
from .kandinsky3 import Kandinsky3
from .koala_1b_llava import Koala1bLlava
from .openjourney import Openjourney
from .pixart512 import PixArt512
from .pixart1024 import PixArt1024
from .playground_25 import Playground25
from .sd_15 import SD15
from .sd_21 import SD21
from .sd_turbo import SDTurbo
from .sdxl import SDXL
from .sdxl_dpo import SDXLDPO
from .sdxl_turbo import SDXLTurbo
from .segmoe import SegMoE
from .ssd1b import SSD1B
from .stable_cascade import StableCascade
from .wurstchen2 import Wurstchen2
from .vega import Vega

generator_name_dict = {
    "amused": Amused,
    "deepfloyd": DeepFloyd,
    "consistency": Consistency,
    "dreamlike-art": DreamlikeArt,
    "dreamlike-photo": DreamlikePhoto,
    "dreamshaper-xl-turbo": DreamshaperXLTurbo,
    "kandinsky2": Kandinsky2,
    "kandinsky3": Kandinsky3,
    "koala-1b-llava": Koala1bLlava,
    "openjourney": Openjourney,
    "pixart-alpha-512": PixArt512,
    "pixart-alpha-1024": PixArt1024,
    "playground-25": Playground25,
    "sdxl": SDXL,
    "sd-15": SD15,
    "sd-21": SD21,
    "sd-turbo": SDTurbo,
    "sdxl-dpo": SDXLDPO,
    "sdxl-turbo": SDXLTurbo,
    "segmoe": SegMoE,
    "ssd1b": SSD1B,
    "stable-cascade": StableCascade,
    "wurstchen2": Wurstchen2,
    "vega": Vega,
}


def get_generator(name: str) -> BaseGenerator:
    name = name.lower()

    generator = generator_name_dict[name]()

    return generator
