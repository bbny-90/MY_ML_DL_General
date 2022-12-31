from typing import Tuple
from abc import (
    ABC, 
    abstractmethod
)
import numpy as np
import scipy
import matplotlib.pyplot as plt

def calc_kernel_rbf(x, y, sig):
    """
        https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    """
    dist = scipy.spatial.distance.cdist(x, y, 'sqeuclidean')
    return np.exp(- 0.5 * dist / sig**2)

class GaussianProcess(ABC):
    def __init__(self, 
        x:np.ndarray, 
        y:np.ndarray, 
        kern_params:dict
        ) -> None:
        self.kern_trn = self.calc_kernel_func(x, x, kern_params)
        self.x_trn = x
        self.y_trn = y
        self.kern_params = kern_params
    
    @abstractmethod
    def calc_kernel_func(self, x, y, kern_params):
        pass

    def infere(self, x:np.ndarray) -> Tuple:
        """
            see page 11:
                An Intuitive Tutorial to Gaussian Processes Regression
                https://arxiv.org/pdf/2009.10862.pdf
        """
        kern_trn_tst = self.calc_kernel_func(self.x_trn, x, self.kern_params)
        kern_tst = self.calc_kernel_func(x, x, self.kern_params)
        s = scipy.linalg.solve(self.kern_trn, kern_trn_tst, assume_a='pos').T
        mean_postr = s @ self.y_trn
        cov_postr = kern_tst - s @ kern_trn_tst
        return mean_postr, cov_postr
    
    def plot(self, x, y, mean_pred, cov_pred):
        std = np.sqrt(np.diag(cov_pred))
        plt.scatter(self.x_trn.flatten(), self.y_trn.flatten(), label='train data')
        plt.plot(x, y, 'r--', label='ground truth')
        plt.plot(x, mean_pred, 'b--', label='mean pred')
        plt.fill_between(x, mean_pred-2.*std, mean_pred+2.*std, color='red', alpha=0.1,)





