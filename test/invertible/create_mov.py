import os
import pathlib
from helper.other import png_to_gif
CODE_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
pjoin = os.path.join

in_dir = pjoin(CODE_PATH, "tmp")
outAdd = pjoin(in_dir, "train.gif")
ids = list(range(0, 3000+20, 20))
png_to_gif(png_dir=in_dir, 
png_prefix="res_", png_ids=ids, gif_add=outAdd, fps=10)