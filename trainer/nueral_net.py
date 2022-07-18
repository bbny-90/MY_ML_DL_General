import time
import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader
from model.nueral_net import MLP


def train_encoder_decoder_mlp(
    encoder: MLP,
    decoder: MLP,
    train_data: numpy.ndarray,
    train_params: dict,
    device: torch.device,
) -> dict:
    """
    train_data: (ndata, nfeat)
                 it should be passed after scaling!
    """
    assert isinstance(train_data, numpy.ndarray)
    assert train_data.ndim == 2
    trainDataPT = TensorDataset(torch.from_numpy(train_data))
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
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    encoder.train()
    decoder.train()
    start_time = time.time()
    loss_report = {"recn_mse": []}
    for epoch in range(num_epochs):
        avg_loss = 0.0
        num_data = 0
        for x in train_loader:
            x = x[0].to(device).float()
            num_data += x.shape[0]
            encoder.zero_grad()
            decoder.zero_grad()
            lat = encoder(x)
            x_ = decoder(lat)
            assert x_.shape == x.shape
            loss = mse(x, x_)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
        avg_loss /= num_data
        loss_report["recn_mse"].append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}", f"Reconst: {avg_loss}")
    print(f"Total Training Time: {time.time() - start_time} seconds")
    return loss_report
