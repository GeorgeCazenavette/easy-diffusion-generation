import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    default="./generated_images",
)
args = parser.parse_args()

prompt_dirs = sorted(os.listdir(args.root_dir))

for prompt_dir in prompt_dirs:
    print(prompt_dir)
    model_dirs = sorted(os.listdir(os.path.join(args.root_dir, prompt_dir)))
    for model_dir in model_dirs:
        num_images = len(
            glob.glob(os.path.join(args.root_dir, prompt_dir, model_dir, "*.png"))
        )
        print("\t{}:\t{}".format(model_dir, num_images))
