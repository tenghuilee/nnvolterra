#%%
r"""
Hack into the MNIST Module
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF

from mnist_module import MNISTClassifier
# from npxconv import outerconv_2d
from plot_utils import savefig
from record_utils import record_append, record_dump_to_file

_arg = argparse.ArgumentParser()
_arg.add_argument("--hack_layer",
                  type=int,
                  default=3,
                  help="hack layer in [1, 2, 3, 4]")
_arg.add_argument("--train_epho",
                  type=int,
                  default=256,
                  help="train epho for hacking network")
_arg.add_argument("--eng", type=float, default=0.4, help="alpha * Ep + X")
args = _arg.parse_args()

HACK_LAYER = args.hack_layer

# read data
# shape N, 28, 28
test_img = np.load("mnist/mnist-test-image.npy")
test_img = test_img.astype(np.float32) / 255.0
assert test_img.shape[1] == 28 and test_img.shape[2] == 28

# shape 6000, int8
test_lab = np.load("mnist/mnist-test-label.npy")

torch_test_img = torch.FloatTensor(test_img.reshape(-1, 1, 28, 28))

train_img = np.load("mnist/mnist-train-image.npy")
train_img = train_img.astype(np.float32) / 255.0
assert train_img.shape[1] == 28 and train_img.shape[2] == 28

#%%


state_dict = torch.load("result/mnist-classifier.torch")
classifier = MNISTClassifier().eval()

classifier.load_state_dict(state_dict)

# accuracy

classifier_preict = classifier.forward(torch_test_img)
classifier_label = torch.argmax(classifier_preict, dim=-1)
classifier_label = classifier_label.data.cpu().numpy()

acc = np.mean(classifier_label == test_lab)

record_append("classifier", "", accuracy=acc)

# compute order one proxy kernels
# the weights
# _W1 = classifier.conv1.weight  # (5, 1, 3, 3)
# _W2 = classifier.conv2.weight  # (10, 5, 3, 3)
# _W3 = classifier.conv3.weight  # (10, 10, 3, 3)
# # compute proxy kernel layer by layer

# # (1, 5, 3, 3) * (5, 10, 3, 3) -> (1, 10, 7, 7) s=4,p=3
# _W = torch.conv_transpose2d(_W1.transpose(0, 1),
#                             _W2.transpose(0, 1),
#                             stride=2,
#                             padding=0)
# # (1, 10, 7, 7) * (10, 10, 3, 3) -> (1, 10, 15, 15) s=8, p=3
# _W = torch.conv_transpose2d(_W, _W3.transpose(0, 1), stride=2, padding=0)
# _W = _W / torch.norm(_W, dim=(2, 3), keepdim=True)
# print(_W.shape)
# exit(0)


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
emuopt = torch.optim.Adam(emu.parameters(), lr=1e-2)

for i in range(args.train_epho):
    np.random.shuffle(train_img)
    for _i in range(1, train_img.shape[0], 64):
        emuopt.zero_grad()
        # _x = torch.rand(64, 1, 28, 28)
        # _x = torch.randn(64, 1 * 28 * 28)
        # _x /= torch.norm(_x, dim=(-1), keepdim=True)
        # _x = _x.reshape(64, 1, 28, 28)
        _x = torch.FloatTensor(train_img[_i:_i + 64].reshape(-1, 1, 28, 28))

        target = classifier.forward(_x, k=HACK_LAYER)
        predict = emu.forward(_x)
        loss = torchF.mse_loss(predict, target)
        loss.backward()
        emuopt.step()

    print(f"\repho {i:4d}/{args.train_epho} loss {loss :.8f}", end="")

print("")

#%%


#%% compute proxy kernel epilson
def compute_ker_perturbation(ker, x, scale):
    """
    ker: shape (n, n)
    x: shape (28, 28)
    """
    f_x = np.fft.fftn(x, axes=(0, 1))
    f_W = np.fft.fftn(ker, s=x.shape, axes=(0, 1))

    Ep = np.fft.ifftn(f_W - f_x, axes=(0, 1)).real
    Ep -= np.min(Ep)
    Ep /= np.max(Ep)

    return scale * Ep + x


# l2 norm of hacker network and classifier network
classifier_pred = classifier.forward(torch_test_img, k=HACK_LAYER)
hacker_pred = emu.forward(torch_test_img)

hacker_label = classifier.forward_rest(hacker_pred, k=HACK_LAYER)

hacker_label = torch.argmax(hacker_label, dim=-1).data.cpu().numpy()

record_append(
    "classifier_hacker",
    "",
    error=torchF.mse_loss(classifier_pred, hacker_pred),
    hacker_accuracy=np.mean(test_lab == hacker_label),
)
# tick_label = [str(_d) for _d in range(10)]
# y = np.linspace(0, 9, 10)
# barhig = 0.35
# y1 = y - barhig / 2
# y2 = y + barhig / 2

predWriter = open(f"result/mnist-hack-{HACK_LAYER}-predict.csv", "w")
# predWriter = sys.stdout  #open(f"result/mnist-hack-{HACK_LAYER}-predict-temp.csv", "w")

predWriter.write("label")
for i in range(10):
    predWriter.write(",%d" % i)
predWriter.write("\n")

# wemu = emu.cv1.weight.data.cpu().numpy()
# wfin = _W.transpose(0, 1).data.cpu().numpy()

# fig, axi = plt.subplots(2, 10, sharex=False, sharey=True)
# for i in range(10):
#     axi[0, i].imshow(wemu[i, 0])
#     axi[0, i].axis("off")
#     axi[1, i].imshow(wfin[i, 0])
#     axi[1, i].axis("off")

# plt.show()

wfin = emu.cv1.weight.data.cpu().numpy()
wfinlen = wfin.shape[0]

for tag in range(wfinlen + 2):

    while True:
        picked_idx = np.random.randint(0, test_img.shape[0])
        obj_x = test_img[picked_idx]
        real_label = test_lab[picked_idx]

        t_obj_x = torch.tensor(obj_x.reshape(1, 1, 28, 28),
                               dtype=torch.float32)
        obj_pred = classifier.forward(t_obj_x)
        print("predict", torch.argmax(obj_pred), real_label)
        # skip wrong prediction
        if torch.argmax(obj_pred[0]) != real_label:
            continue
        break

    if tag < wfinlen:
        weig = wfin[tag, 0].copy()
    elif tag == wfinlen:
        weig = np.random.rand(wfin.shape[2], wfin.shape[3])
    else:
        weig = np.random.randn(wfin.shape[2], wfin.shape[3])

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(obj_x) #, cmap='gray')
    plt.axis('off')
    savefig(fig, f"mnist-hack-{HACK_LAYER}-img-{tag}")
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(weig) #, cmap='gray')
    plt.axis('off')
    savefig(fig, f"mnist-hack-{HACK_LAYER}-patch-{tag}")
    # plt.show()
    plt.close()

    # Epx = randX.copy()
    # Epx[0:weig.shape[0],0:weig.shape[1]] = weig

    Epx = compute_ker_perturbation(weig, obj_x, args.eng)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(Epx)  #, cmap='gray')
    plt.axis('off')
    savefig(fig, f"mnist-hack-{HACK_LAYER}-img-patch-{tag}")
    # plt.show()
    plt.close()

    # prepate data
    t_Epx = torch.tensor(Epx.reshape(1, 1, 28, 28), dtype=torch.float32)

    pke_pred = classifier.forward(t_Epx)

    # print(torch.topk(pke_pred, k=3)[1])
    obj_pred = obj_pred[0].data.cpu().numpy()
    pke_pred = pke_pred[0].data.cpu().numpy()

    record_append(
        f"hack-{tag}",
        "",
        label=real_label,
        hack_label=np.argmax(pke_pred),
        object_predict=obj_pred,
        patched_predict=pke_pred,
    )

    predWriter.write("preX %d" % tag)
    for x in obj_pred:
        predWriter.write(",%.8f" % x)
    predWriter.write("\npreE %d" % tag)
    for x in pke_pred:
        predWriter.write(",%.8f" % x)
    predWriter.write("\n")

print("all done")

predWriter.close()

record_dump_to_file(f"mnist-hack-{HACK_LAYER}-predict")
# record_dump_to_file(sys.stdout)
