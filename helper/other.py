import os


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