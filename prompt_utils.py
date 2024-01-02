import os
import random
import glob

prompt_files = glob.glob("./prompts/*.txt")
prompt_path_dict = {
    os.path.basename(f)[:-4]: f for f in prompt_files
}

def load_prompts(name: str, shuffle: bool = False, max_images: int = None) -> list[str]:
    try:
        path = prompt_path_dict[name.lower()]
    except:

        raise FileNotFoundError(
            'Prompt file not found at `{}`.'.format(
                path
            )
        )

    with open(path) as f:
        prompt_list = f.read().splitlines()

    if shuffle:
        random.shuffle(prompt_list)

    if max_images is not None:
        prompt_list = prompt_list[:max_images]

    return prompt_list
