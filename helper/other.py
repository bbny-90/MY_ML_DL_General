import os
from typing import List
import os
import imageio
pjoin = os.path.join


def drop_file_type(filename: str, filetype: str):
    if filename.endswith("." + filetype):
        filename_ = filename.split("." + filetype)[0]
    else:
        filename_ = filename
    return filename_


def make_dirs(add):
    if os.path.exists(add):
        os.system(f"rm -r {add}")
    os.makedirs(add)

def png_to_gif(
    png_dir:str, 
    png_prefix:str, 
    png_ids:List[int], 
    gif_add:str,
    fps=5
    ) -> None:
    images = []
    # if os.path.isdir(gif_dir):
    #     rmtree(gif_dir)
    # os.mkdir(gif_dir)
        
    # for file_name in os.listdir(png_dir):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(png_dir, file_name)
    #         images.append(imageio.imread(file_path))
    for i in png_ids:
        file_path = pjoin(png_dir, f"{png_prefix}{i}.png")
        print(file_path)
        images.append(imageio.imread(file_path))
    print(fps)
    imageio.mimwrite(gif_add, images, fps=fps, subrectangles=True)