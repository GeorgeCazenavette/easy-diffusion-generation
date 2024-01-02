import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from PIL import Image

import generators
import options
import prompt_utils

args = options.parse_args()

prompts = prompt_utils.load_prompts(args.prompts, args.shuffle_prompts, args.max_images)

gen = generators.get_generator(args.model)

if args.inference_steps is not None:
    gen.num_inference_steps = args.inference_steps
    args.model += "_{}-steps".format(args.inference_steps)

if args.det_seed:
    args.model += "_det-seed"

save_dir = os.path.join(args.save_root, args.prompts, args.model)
os.makedirs(save_dir, exist_ok=True)

for i, p in enumerate(prompts):
    save_path = os.path.join(save_dir, "{:05}.png".format(i))

    if args.skip_existing and os.path.exists(save_path):
        continue

    if args.det_seed:
        g = torch.Generator(device="cuda").manual_seed(i)
    else:
        g = None

    with torch.no_grad():
        im = gen(p, g)

    im.save(save_path)
