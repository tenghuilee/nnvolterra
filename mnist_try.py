r"""
Try MNIST trained module from mnist_net.py
"""

import torch
import numpy as np
from mnist_module import MNISTClassifier

import os


# shape N, 28, 28
train_img = np.load(
    os.path.expanduser("mnist/mnist-test-image.npy")
)
assert train_img.shape[1] == 28 and train_img.shape[2] == 28

# shape N, int8
train_lab = np.load(
    os.path.expanduser("mnist/mnist-test-label.npy")
)

timg = torch.tensor(train_img.reshape(-1,1,28,28)/255.0, dtype=torch.float32)
tlab = torch.tensor(train_lab, dtype=torch.long)

module = MNISTClassifier()
module.load_state_dict(torch.load("result/mnist-classifier.torch", map_location="cpu"))

predict = module.forward(timg)

_, midx = torch.max(predict, dim=1)

print(midx)
print(tlab)

print(torch.sum(midx == tlab)/train_img.shape[0])

