# a simple MNIST classifier for hacker

import torch
import torch.nn.functional as torchF
import numpy as np
from mnist_module import MNISTClassifier 

import logging


logger = logging.getLogger("mnist")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter(
    "%(asctime)s %(filename)s:%(lineno)d[%(levelname)s]%(message)s")
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_sh)

logger.info("MNIST Start")

# load data
logger.info("loading training data")
# shape 60000, 28, 28, int8
train_img = np.load("mnist/mnist-train-image.npy")
train_img = train_img.astype(np.float32) / 255.0
assert train_img.shape[1] == 28 and train_img.shape[2] == 28

# shape 6000, int8
train_lab = np.load("mnist/mnist-train-label.npy")


def data_iter(batchsize):
    """yield img,label in torch.Tensor"""

    count = 0

    numimg = train_img.shape[0]

    idx = np.random.permutation(numimg)

    for i in range(0, numimg, batchsize):
        img = train_img[idx[i:i + batchsize]]
        lab = train_lab[idx[i:i + batchsize]]

        # img = img.astype(np.float32)

        timg = torch.tensor(img.reshape(-1, 1, 28, 28), dtype=torch.float32)
        tlab = torch.tensor(lab, dtype=torch.int64)

        yield count, timg, tlab
        count += 1


# the network
classifier = MNISTClassifier()
CHECK_POINT = "result/mnist-classifier.torch"

optim = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

# start train
logger.info("start train and save to %s", CHECK_POINT)
for e in range(512):
    for i, img, lab in data_iter(128):
        optim.zero_grad()
        randval = np.random.rand()
        if randval < 0.1:
            pred = classifier.forward(img + torch.randn_like(img).mul_(0.2))
        elif randval < 0.2:
            pred = classifier.forward(img + torch.rand_like(img).mul_(0.2))
        else:
            pred = classifier.forward(img)

        loss = torchF.cross_entropy(pred, lab)
        loss.backward()
        optim.step()

        if i % 64 == 0:
            logger.info("%03d-%05d loss %.8f", e, i, loss.data.cpu().numpy())

    torch.save(classifier.state_dict(), CHECK_POINT)
