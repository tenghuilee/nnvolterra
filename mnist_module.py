r"""
A simple MNIST Classifier module
"""

import torch
import torch.nn as nn
import torch.nn.functional as torchF

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, 3, 2, 1)
        self.conv2 = nn.Conv2d(10, 10, 3, 2, 1)
        self.conv3 = nn.Conv2d(10, 10, 3, 2)
        self.conv4 = nn.Conv2d(10, 10, 3, 1)

        self.lin1 = nn.Linear(10, 10)

        # self.act = torch.relu
        self.act = torch.sigmoid

    def forward(self, input: torch.Tensor, k=-1) -> torch.Tensor:
        out = self.conv1(input)
        out = self.act(out)
        if k == 1:
            return out
        out = self.conv2(out)
        out = self.act(out)
        if k == 2:
            return out
        out = self.conv3(out)
        out = self.act(out)
        if k == 3:
            return out
        out = self.conv4(out)
        out = self.forward_fc(out)
        return out
    
    def forward_rest(self, input: torch.Tensor, k=-1) -> torch.Tensor:
        out = input
        if k == 1:
            # out = self.act(out)
            out = self.conv2(out)
            out = self.act(out)
            out = self.conv3(out)
            out = self.act(out)
            out = self.conv4(out)
        elif k == 2:
            # out = self.act(out)
            out = self.conv3(out)
            out = self.act(out)
            out = self.conv4(out)
        elif k >= 3:
            # out = self.act(out)
            out = self.conv4(out)

        out = self.forward_fc(out)
        return out
    
    def forward_fc(self, input: torch.Tensor) -> torch.Tensor:
        # input shape (N, 10, 1, 1)
        out = torch.mean(input, dim=(2,3))
        out = self.act(out)
        out = self.lin1(out)
        return out

