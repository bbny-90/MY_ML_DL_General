import os
import pathlib
import numpy as np
import torch
from torch import (
    optim, 
    distributions
)
from sklearn import datasets

from model.invertible.mlp import RealNVP
from model.invertible.train import train

import matplotlib.pyplot as plt
from pylab import rcParams
CODE_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
pjoin = os.path.join
torch.manual_seed(2022)
np.random.seed(2022)
rcParams['figure.figsize'] = 10, 8

BATCH_SIZE = 128
EPOCHS = 20
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIM = 256
N_COUPLING_LAYERS = 6
N_TRAIN_SAMPLES = int(30e3)
N_TEST_SAMPLES = int(1e3)
NOISE_LEVEL = 0.05 
OUT_DIR = pjoin(CODE_PATH, "tmp")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def subfig_plot(
    ax,
    data, 
    x_start, 
    x_end, 
    y_start, 
    y_end, 
    title, 
    color, 
    x_lines=[], 
    ):
    for l in x_lines:
        ax.plot(l[:, 0], l[:, 1], c='k', alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], c=color, s=1)
    ax.set_xlim(x_start,x_end)
    ax.set_ylim(y_start,y_end)
    ax.set_title(title)

def create_plot_lines(x_start, x_end, y_start, y_end, nx, ny):
    lines = []
    y = np.linspace(y_start, y_end, ny)
    for x in np.linspace(x_start, x_end, nx):
        lines.append(
            np.stack([x*np.ones_like(y), y], axis=1)
        )
    x = np.linspace(x_start, x_end, nx)
    for y in np.linspace(y_start, y_end, ny):
        lines.append(
            np.stack([x, y*np.ones_like(x)], axis=1)
        )
    return lines

def plot_test(x_test, z_test, x_sample, z_sample, itr):
    fig, ax = plt.subplots(2, 2)
    x_lines = create_plot_lines(-2, 3, -1, 1.5, 20, 20)
    subfig_plot(ax[0][0], x_test, -2, 3, -1, 1.5, r'real space: ' +r'$x \sim p(x)$', 'b', x_lines=x_lines)
    z_lines = []
    for l in x_lines:
        z_lines.append(
            model(torch.from_numpy(l.astype(np.float32)))[0].numpy()
        )
    subfig_plot(ax[0][1], z_test, -3, 3, -3,3,r'latent space: ' + r'$z = f(x)$', 'r', x_lines = z_lines)
    z_lines = create_plot_lines(-3, 3, -3, 3, 20, 20)
    x_lines = []
    for l in z_lines:
        x_lines.append(
            model.inverse(torch.from_numpy(l.astype(np.float32))).numpy()
        )
    subfig_plot(ax[1][0], z_sample, -3, 3, -3, 3, r'latent space: ' + r'$z \sim p(z)$', 'r', x_lines=z_lines)
    subfig_plot(ax[1][1], x_sample, -2, 3, -1, 1.5,r'real space: ' + r'$x = g(z)$', 'b', x_lines=x_lines)
    fig.suptitle(f"iteration: {itr}")
    plt.savefig(pjoin(OUT_DIR, f"res_{itr}.png"))

@torch.no_grad()
def test(model, test_loader, itr):
    model.eval()
    x_test = next(iter(test_loader))
    z_test, _ = model(x_test)
    x_sample, z_sample = model.sample(num_sample=N_TEST_SAMPLES)
    plot_test(x_test, z_test, x_sample, z_sample, itr)

# --- data loading --- #
train_data = datasets.make_moons(n_samples=N_TRAIN_SAMPLES, noise=NOISE_LEVEL)[0].astype(np.float32)
test_data = datasets.make_moons(n_samples=N_TEST_SAMPLES, noise=NOISE_LEVEL)[0].astype(np.float32)

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = N_TEST_SAMPLES,shuffle=True, **kwargs
)
mask_templ = np.array([0, 1]).astype(np.float32)
mask = np.array([0, 1]).astype(np.float32)
prior_z = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

model = RealNVP(
        prior_z=prior_z,
        input_dim = INPUT_DIM,
        output_dim = OUTPUT_DIM,
        hid_dim = HIDDEN_DIM,
        mask = mask,
        num_coupl_layers = N_COUPLING_LAYERS,
        num_mlp_hid_layers=3,
        act_type = 'relu'
)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
cntr = 0
for epoch in range(EPOCHS):
    train_loss = 0.
    for batch_idx, data in enumerate(train_loader):
        model.train()
        btch_loss = train(
            data=data,
            model=model, 
            optimizer=optimizer
        )
        train_loss += btch_loss
        if cntr % 20 == 0:
            test(model, test_loader, cntr)
        cntr += 1
    print(epoch, train_loss)