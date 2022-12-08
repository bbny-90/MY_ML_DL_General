import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self,
        space_dim_input = 28,
        num_out_classes = 10,
        input_channel=1,
        kernel_size=5,
        stride=1,
        hidden_channels = [16, 16]
        ):
        super(CNN2D, self).__init__()
        assert (kernel_size - stride) % 2 == 0
        hidden_channels = [input_channel] + hidden_channels
        conv = []
        for i in range(len(hidden_channels) - 1):
            assert space_dim_input % 2 == 0
            conv.append(
                nn.Conv2d(
                    in_channels=hidden_channels[i],              
                    out_channels=hidden_channels[i+1],            
                    kernel_size=kernel_size,              
                    stride=stride,                   
                    padding = (kernel_size - stride) // 2,# leads to the same output dim as input
                ),
            )
            # keep input spatioal dim
            conv.append(nn.ReLU())
            conv.append(nn.MaxPool2d(kernel_size=2))
            # make spatial dim half
            space_dim_input //= 2
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Linear(space_dim_input*space_dim_input*hidden_channels[-1], num_out_classes)
    
    def forward(self, x):
        """
            x: (bsz, ch, nx, ny)
        """
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        score = self.fc(x)
        return score


def train(optim, model, x, label, loss_func):
    model.train()
    pred = model(x)
    loss = loss_func(pred, label)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()

@torch.no_grad()
def test(model, x, y):
    model.eval()
    test_output = model(x)
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    accuracy = (pred_y == y).sum().item() / float(y.size(0))
    return accuracy