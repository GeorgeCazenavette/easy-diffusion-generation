import argparse
import glob
import os
import tqdm

import dominate
from dominate.tags import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    default="./generated_images",
)
args = parser.parse_args()

prompt_files = glob.glob(os.path.join("prompts", "*.txt"))

prompt_path_dict = {
    os.path.basename(p)[:-4]: p for p in prompt_files
}

class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir)
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(
            title=title, style="color: white; background-color: #202124"
        )
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh), charset="utf-8")
        else:
            with self.doc.head:
                meta(charset="utf-8")

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h1(str, style="text-align:center")

    def add_images(self, ims, txts, links, width=400, row=5):
        # self.add_table()
        zipped = list(zip(ims, txts, links))

        with self.doc:
            with div(style="text-align:center"):
                for im, txt, link in zipped:
                    with div(
                        style="text-align:center;width:256;display: inline-block;vertical-align:top;"
                    ):
                        with a(href=os.path.join(link), id=im.split("/")[-1][:-4]):
                            img(
                                width="256px",
                                height="256px",
                                src=os.path.join(im),
                                loading="lazy",
                                style="object-fit: cover;",
                            )
                        br()
                        p(
                            txt,
                            style="text-align:center; display:inline-block; width: 256px; margin: 5px",
                        )

    def save(self):
        html_file = "%s/index.html" % self.web_dir
        f = open(html_file, "wt")
        f.write(self.doc.render())
        f.close()


leaf_dirs = []

for dirpath, dirnames, filenames in os.walk(args.root_dir):
    if not dirnames:
        leaf_dirs.append(os.path.abspath(dirpath))

for ld in tqdm.tqdm(leaf_dirs):
    path_parts = ld.split(os.sep)

    title = os.path.basename(ld)
    prompt_set = path_parts[-2]

    prompt_path = prompt_path_dict[prompt_set]

    with open(prompt_path) as f:
        caption_list = f.read().splitlines()

    html = HTML(ld, "images")
    html.add_header("{} ({})".format(title, prompt_set.replace("_", " ")))

    files = sorted(glob.glob(ld + "/**/*.png", recursive=True))

    ims = []
    txts = []
    links = []

    for i, c in enumerate(caption_list):
        f = "{:05}.png".format(i)
        ims.append(f)
        txts.append(c)
        links.append(f)

    html.add_images(ims, txts, links, width=256, row=5)
    html.save()
