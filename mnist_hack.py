#%%
r"""
Hack into the MNIST Module
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
# from npxconv import outerconv_2d
import itertools

import matplotlib.pyplot as plt
from mnist_module import MNISTClassifier

import argparse
_arg = argparse.ArgumentParser()
_arg.add_argument("--hack_layer", type=int, default=3, help="hack layer in [1, 2, 3, 4]")
args = _arg.parse_args()

HACK_LAYER = args.hack_layer

def savefig(fig, tag):
    print("save tag", tag)
    # dpi: 160 240 320 460 640 
    fig.savefig(f"result/{tag}.eps", dpi=160, bbox_inches='tight', transparent=True, pad_inches=0)
    # fig.savefig(f"result/{tag}.pdf", dpi=600, bbox_inches='tight', transparent=True, pad_inches=0)
    # fig.savefig(f"result/{tag}.png", dpi=600, bbox_inches='tight', transparent=True, pad_inches=0)
    fig.savefig(f"result/{tag}.jpg", dpi=160, bbox_inches='tight', transparent=True, pad_inches=0)

#%%

state_dict = torch.load("result/mnist-classifier.torch")

module = MNISTClassifier().eval()
module.load_state_dict(state_dict)


# emulate order1
class EMUo1(nn.Module):
    def __init__(self):
        super().__init__()
        if HACK_LAYER == 1:
            self.cv1 = nn.Conv2d(1, 5, 3, 2, 1)
        elif HACK_LAYER == 2:
            self.cv1 = nn.Conv2d(1, 10, 7, 4, 3)
        elif HACK_LAYER == 3:
            self.cv1 = nn.Conv2d(1, 10, 15, 8, 3)
        elif HACK_LAYER == 4:
            self.cv1 = nn.Conv2d(1, 10, 31, 8, 3)
    
    def forward(self, input):
        out = self.cv1(input)
        if HACK_LAYER == 4:
            out = out.reshape(out.shape[0], out.shape[1])
        return out

#%%

emu = EMUo1()
emuopt = torch.optim.Adam(emu.parameters(), lr=2e-3)

for i in range(500):
    emuopt.zero_grad()
    _x = torch.rand(128,1,28,28)
    target = module.forward(_x, k=HACK_LAYER)
    predict = emu.forward(_x)
    loss = torchF.mse_loss(predict, target)
    loss.backward()
    emuopt.step()

    print(f"\repho {i:4d} loss {loss :.8f}", end="")

print("")

#%%

wfin = emu.cv1.weight.data.cpu().numpy()

# read data
# shape N, 28, 28
test_img = np.load(
    os.path.expanduser("mnist/mnist-test-image.npy")
) / 255.0
assert test_img.shape[1] == 28 and test_img.shape[2] == 28

# tick_label = [str(_d) for _d in range(10)]
# y = np.linspace(0, 9, 10)
# barhig = 0.35
# y1 = y - barhig / 2
# y2 = y + barhig / 2

predWriter = open(f"result/mnist-hack-{HACK_LAYER}-predict.csv", "w")

predWriter.write("label")
for i in range(10):
    predWriter.write(",%d"%i)
predWriter.write("\n")

wfinlen = wfin.shape[0]

for tag in range(wfinlen + 2):

    randX = test_img[np.random.randint(0, test_img.shape[0])]

    if tag < wfinlen:
        weig = wfin[tag,0].copy()
    elif tag == wfinlen:
        weig = np.random.rand(wfin.shape[2], wfin.shape[3])
    else:
        weig = np.random.randn(wfin.shape[2], wfin.shape[3])

    fig = plt.figure(figsize=(8,8))
    plt.imshow(weig)#, cmap='gray')
    plt.axis('off')
    savefig(fig, f"mnist-hack-{HACK_LAYER}-patch-{tag}")
    plt.close()


    # Epx = randX.copy()
    # Epx[0:weig.shape[0],0:weig.shape[1]] = weig

    fRandX = np.fft.fftn(randX, axes=(0, 1))
    fWf1 = np.fft.fftn(weig, s=randX.shape, axes=(0, 1))

    Ep = np.fft.ifftn(fWf1 - fRandX, axes=(0, 1)).real
    Ep = Ep - np.min(Ep)
    Ep = Ep / np.max(Ep)

    Epx = 0.35 * Ep + randX

    predX = module.forward(
        torch.tensor(randX.reshape(1, 1, 28, 28), dtype=torch.float32))

    predE = module.forward(
        torch.tensor(Epx.reshape(1, 1, 28, 28), dtype=torch.float32))

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(Epx)#, cmap='gray')
    plt.axis('off')
    savefig(fig, f"mnist-hack-{HACK_LAYER}-img-{tag}")
    plt.close()

    predX = predX[0].data.cpu().numpy()
    predE = predE[0].data.cpu().numpy()

    # fig = plt.figure(figsize=(8, 4))
    # plt.barh(y1, predX, height=barhig)
    # plt.barh(y2, predE, height=barhig)
    # plt.yticks(y, tick_label)
    # plt.legend(["predict", "patch added"])
    # savefig(fig, f"mnist-hack-predict-{tag}")
    # plt.close()

    predWriter.write("preX %d"%tag)
    for x in predX:
        predWriter.write(",%.8f"%x)
    predWriter.write("\npreE %d"%tag)
    for x in predE:
        predWriter.write(",%.8f"%x)
    predWriter.write("\n")

print("all done")

predWriter.close()
