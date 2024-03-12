import os, requests

def get_parti_prompts():
    url = "https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv"
    response = requests.get(url)

    if response.status_code == 200:
        lines = str(response.content.decode()).splitlines()[1:]

    else:
        print('Failed to download Parti')
        return

    prompts = [l.split("\t")[0] for l in lines]

    with open(os.path.join("prompts", "parti.txt"), "w") as f:
        f.write("\n".join(prompts))

def get_t2i_comp_bench_val():
    all_prompts = []

    categories = ["color", "complex", "non_spatial", "shape", "texture"]
    for c in categories:
        url = "https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/main/examples/dataset/{}_val.txt".format(c)
        response = requests.get(url)
        if response.status_code == 200:
            prompts = str(response.content.decode()).splitlines()

        else:
            print("Failed to download T2I CompBench {}".format(c))
            continue

        with open(os.path.join("prompts", "t2i_comp_bench_{}.txt".format(c)), "w") as f:
            f.write("\n".join(prompts))

        all_prompts += prompts

    with open(os.path.join("prompts", "t2i_comp_bench_all.txt"), "w") as f:
        f.write("\n".join(all_prompts))


def get_dalle3():
    url = "https://raw.githubusercontent.com/openai/dalle3-eval-samples/main/prompts/dalle3_eval.txt"
    response = requests.get(url)

    if response.status_code == 200:
        prompts = str(response.content.decode()).splitlines()
    else:
        print("Failed to download DALL-E 3 Prompts")
        return

    with open(os.path.join("prompts", "dalle3.txt"), "w") as f:
        f.write("\n".join(prompts))

def get_drawbench():
    url = "https://raw.githubusercontent.com/openai/dalle3-eval-samples/main/prompts/drawbench.txt"
    response = requests.get(url)

    if response.status_code == 200:
        prompts = str(response.content.decode()).splitlines()
    else:
        print("Failed to download DrawBench Prompts")
        return

    with open(os.path.join("prompts", "drawbench.txt"), "w") as f:
        f.write("\n".join(prompts))

def get_coco():
    url = "https://raw.githubusercontent.com/Schuture/Benchmarking-Awesome-Diffusion-Models/main/Prompts/COCO_Captions.csv"
    response = requests.get(url)

    if response.status_code == 200:
        prompts = str(response.content.decode()).splitlines()
    else:
        print("Failed to download COCO Prompts")
        return

    with open(os.path.join("prompts", "coco.txt"), "w") as f:
        f.write("\n".join(prompts))


if __name__ == "__main__":
    get_parti_prompts()
    get_t2i_comp_bench_val()
    get_dalle3()
    get_coco()