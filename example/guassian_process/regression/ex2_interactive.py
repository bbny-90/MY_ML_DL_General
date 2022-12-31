import numpy as np
import matplotlib.pyplot as plt
from main import (
    GaussianProcess, 
    calc_kernel_rbf
)

class MYGP(GaussianProcess):
    def __init__(self, x: np.ndarray, y: np.ndarray, kern_params: dict) -> None:
        super().__init__(x, y, kern_params)
    
    def calc_kernel_func(self, x, y, kern_params):
        return calc_kernel_rbf(x, y, kern_params['sig'])



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

all_points = []
def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        (event.button, event.x, event.y, event.xdata, event.ydata))
    all_points.append([event.xdata, event.ydata])
    plt.plot(event.xdata, event.ydata, 'ro')
    fig.canvas.draw()

def onpress(event):
    plt.clf()
    tmp = np.array(all_points)
    gp = MYGP(tmp[:, 0:1], tmp[:, 1:2], {'sig':1.})
    xtst = np.linspace(0., 10., 50).reshape(-1, 1)
    mean_tst, cov_tst = gp.infere(xtst)
    xtst = xtst.flatten()
    mean_tst = mean_tst.flatten()

    plt.scatter(tmp[:, 0], tmp[:, 1])
    plt.scatter(tmp[:, 0], tmp[:, 1])
    std = np.sqrt(np.diag(cov_tst))
    plt.plot(xtst, mean_tst, 'b--')
    plt.fill_between(xtst, mean_tst-2.*std, mean_tst+2.*std, color='red', alpha=0.1,)

    fig.canvas.draw()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', onpress)
plt.show()
print(all_points)