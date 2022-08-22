import time
import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader
# from model.nueral_net import MLP
from model.autoencoder_pt import AEBase


def train_encoder_decoder(
    ae_model: AEBase,
    train_data: numpy.ndarray,
    train_params: dict,
    device: torch.device,
) -> dict:
    """
    train_data: (ndata, nfeat)
                 it should be passed after scaling!
    """
    assert isinstance(train_data, numpy.ndarray)
    # assert train_data.ndim == 3
    trainDataPT = TensorDataset(torch.from_numpy(train_data))
    train_loader = DataLoader(
        trainDataPT, shuffle=True, batch_size=train_params["batchsize"], drop_last=False
    )
    lr: float = train_params["lr"]
    num_epochs: int = train_params["epochs"]
    mse = torch.nn.MSELoss()
    if train_params["optimizer"] == "ADAM":
        optimizer = torch.optim.Adam
    elif train_params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD
    else:
        raise NotImplementedError(
            train_params["optimizer"]
        )  # TODO: needs better err msg
    optimizer = optimizer(
        list(ae_model.encoder.parameters()) + list(ae_model.decoder.parameters()), lr=lr
    )
    ae_model.encoder.train()
    ae_model.decoder.train()
    start_time = time.time()
    loss_report = {"recn_mse_train": []}
    for epoch in range(num_epochs):
        avg_loss = 0.0
        num_data = 0
        for x in train_loader:
            x = x[0].to(device).float()
            num_data += x.shape[0]
            ae_model.encoder.zero_grad()
            ae_model.decoder.zero_grad()
            lat = ae_model.encode(x)
            x_ = ae_model.decode(lat)
            assert x_.shape == x.shape
            loss = mse(x, x_)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
        avg_loss /= num_data
        loss_report["recn_mse_train"].append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}", f"Reconst: {avg_loss}")
    print(f"Total Training Time: {time.time() - start_time} seconds")
    return loss_report


def train_forcast(
    model: torch.nn.Module,
    x_train: numpy.ndarray,
    y_train: numpy.ndarray,
    train_params: dict,
    device: torch.device,
) -> dict:
    """
    train_data: (ndata, nfeat)
                 it should be passed after scaling!
    """
    assert isinstance(x_train, numpy.ndarray)
    assert isinstance(y_train, numpy.ndarray)
    assert x_train.ndim == 3
    assert y_train.ndim == 3
    x_trainDataPT = TensorDataset(torch.from_numpy(x_train))
    train_loader = DataLoader(
        trainDataPT, shuffle=True, batch_size=train_params["batchsize"], drop_last=False
    )
    lr: float = train_params["lr"]
    num_epochs: int = train_params["epochs"]
    mse = torch.nn.MSELoss()
    if train_params["optimizer"] == "ADAM":
        optimizer = torch.optim.Adam
    else:
        raise NotImplementedError(
            train_params["optimizer"]
        )  # TODO: needs better err msg
    optimizer = optimizer(
        list(ae_model.encoder.parameters()) + list(ae_model.decoder.parameters()), lr=lr
    )
    ae_model.encoder.train()
    ae_model.decoder.train()
    start_time = time.time()
    loss_report = {"recn_mse_train": []}
    for epoch in range(num_epochs):
        avg_loss = 0.0
        num_data = 0
        for x in train_loader:
            x = x[0].to(device).float()
            num_data += x.shape[0]
            ae_model.encoder.zero_grad()
            ae_model.decoder.zero_grad()
            lat = ae_model.encode(x)
            x_ = ae_model.decode(lat)
            assert x_.shape == x.shape
            loss = mse(x, x_)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
        avg_loss /= num_data
        loss_report["recn_mse_train"].append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}", f"Reconst: {avg_loss}")
    print(f"Total Training Time: {time.time() - start_time} seconds")
    return loss_report