r"""
A simple MNIST Classifier module
"""

import torch
import torch.nn as nn
import torch.nn.functional as torchF

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 5, 3, 2, 1)
        self.conv2 = nn.Conv2d(5, 10, 3, 2, 1)
        self.conv3 = nn.Conv2d(10, 10, 3, 2)
        self.conv4 = nn.Conv2d(10, 10, 3, 1)

    def forward(self, input: torch.Tensor, k=-1) -> torch.Tensor:
        out = self.conv1(input)
        # out = torch.relu_(out)
        out = torch.sigmoid_(out)
        if k == 1:
            return out
        out = self.conv2(out)
        # out = torch.relu_(out)
        out = torch.sigmoid_(out)
        if k == 2:
            return out
        out = self.conv3(out)
        # out = torch.relu_(out)
        out = torch.sigmoid_(out)
        if k == 3:
            return out
        out = self.conv4(out)
        out = torch.reshape(out, (-1, 10))
        return out
