import os
import sys
import pathlib
import inspect

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROB_FILE_GEN = os.path.join(DIR_PATH, "..")
sys.path.append(PROB_FILE_GEN)

def test1():
    import numpy as np
    from helper.data_processor import create_window_data
    x = np.random.rand(20, 3)
    w_size = 3
    y = create_window_data(x, w_size, 2)
    assert y.shape[1:] == (w_size * x.shape[1], ), y.shape[1:]
    y = create_window_data(x, w_size, 3)
    assert y.shape[1:] == (w_size, x.shape[1])
    print(f"{inspect.stack()[0][3]} is passed")

def test2():
    import numpy as np
    from helper.data_processor import create_window_forcast_data
    data = np.random.rand(20, 3)
    w_size, num_future = 3, 1
    x, y = create_window_forcast_data(data, w_size, num_future)
    assert y.shape[1:] == (num_future, x.shape[1]), y.shape
    assert x.shape[1:] == (w_size, x.shape[1])
    print(f"{inspect.stack()[0][3]} is passed")


if __name__ == "__main__":
    test1()
    test2()