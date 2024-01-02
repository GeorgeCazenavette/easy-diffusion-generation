import argparse

from generators import generator_name_dict
from prompt_utils import prompt_path_dict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="consistency",
        choices=list(generator_name_dict.keys()),
        help="Which diffusion model to use.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help="Uses per-model settings by default.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="diffusiondb",
        help="list of prompts. pre-registered are {}".format(
            list(prompt_path_dict.keys())
        ),
    )
    parser.add_argument(
        "--shuffle_prompts", type=bool, default=False, help="Shuffle the prompt list?"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum numbers of image to generate. Default is length of prompt list.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./generated_images",
        help="Where to save images",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="whether to skip images that already exist",
    )

    parser.add_argument("--det_seed", action="store_true")

    args = parser.parse_args()

    return args
