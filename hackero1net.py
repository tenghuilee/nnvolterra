import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF


class HackerO1Net1D(nn.Module):

    def __init__(self, kernel_size) -> None:
        super().__init__()

        self.cv1 = nn.Conv1d(1, 1, kernel_size, 1, 0)

    def forward(self, input: torch.Tensor):
        return self.cv1(input)

    @property
    def order0(self):
        return self.cv1.bias.squeeze().detach().numpy()

    @property
    def order1(self):
        ans = self.cv1.weight.squeeze().detach().numpy()
        return np.flip(ans).copy()

    def fit(self, func: callable, numiter=1000, err=1e-8, verbose=False):
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        for i in range(numiter):
            optim.zero_grad()

            # random x
            train_x = torch.randn(16, 1, 512)
            train_x /= torch.norm(train_x, dim=-1, keepdim=True)

            # compute g * sigmoid(h * x)
            label_x = func(train_x)
            estim_x = self.forward(train_x)
            loss = torchF.mse_loss(label_x, estim_x)
            loss.backward()
            optim.step()

            if loss < err:
                break
            if verbose:
                print(f"\repho {i}, loss {loss :.8f}", end="")
        if verbose:
            print()
