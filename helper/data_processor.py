from typing import (Tuple,
                    List)
import numpy
from numpy import ndarray

def __2d_to_2d(
    data: ndarray,
    w_size: int,
) -> ndarray:
    out = []
    i = 0
    while i + w_size <= data.shape[0]:
        out.append(data[i:i+w_size, :].flatten())
        i += 1
    out = numpy.stack(out, axis=0)
    return out

def __2d_to_3d(
    data: ndarray,
    w_size: int,
) -> ndarray:
    out = []
    i = 0
    while i + w_size <= data.shape[0]:
        out.append(data[i:i+w_size, :])
        i += 1
    out = numpy.stack(out, axis=0)
    return out

def create_window_data(
    data: ndarray,
    w_size: int,
    out_ndim: int,
    ) -> ndarray:
    assert data.ndim == 2
    if out_ndim == 2:
        return __2d_to_2d(data, w_size)
    elif out_ndim == 3:
        return __2d_to_3d(data, w_size)
    else:
        raise NotImplementedError(
            f"output dim {out_ndim} is not supported."
            )

def create_window_forcast_data(
    data: ndarray,
    w_size: int,
    num_future: int,
    out_ndim = 3,
    ) -> Tuple[ndarray, ndarray]:
    """
        data: (m, n)
        x   : (b, w_size, n)
        y   : (b, num_future, n)
    """
    assert data.ndim == 2
    assert out_ndim == 3 # 2 would be compatible with MLP
    x, y = [], []
    i = 0
    while i+w_size+num_future <= data.shape[0]:
        x.append(data[i:i+w_size, :])
        y.append(data[i+w_size:i+w_size+num_future, :])
        i += 1
    x = numpy.stack(x, axis=0)
    y = numpy.stack(y, axis=0)
    return x, y

def shuffle(
    data:List[ndarray],
    inplace = True,
):
    n = data[0].shape[0]
    for i, d in enumerate(data):
        assert d.shape[0] == n, f"{i}th data has shape {d.shape}"
    shuff = numpy.arange(n)
    numpy.random.shuffle(shuff)
    if inplace:
        data = [d[shuff,...] for d in data]
        return None
    else:
        return [d[shuff,...] for d in data]


def split_data(
    data:ndarray,
    ratio: float
    ) -> Tuple[ndarray, ndarray]:
    """
        data: with any ndim
    """
    assert 0.< ratio < 1., f"{ratio} must be in (0., 1)"
    data_ = numpy.copy(data)
    ind = int(ratio * data.shape[0])
    return data_[:ind, ...], data_[ind:, ...]
    