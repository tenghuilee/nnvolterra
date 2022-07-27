#!/usr/bin/env python
# coding: utf-8

# # Rank for outer convolution And Check For Volterra

import argparse
from sys import stdout

import numpy as np
import torch

import tensordec
from conv_zero import *
from hackero1net import HackerO1Net1D
from record_utils import record_append, record_dump_to_file
from npxconv import conv_ordern, outerconv, outerconv_diag

_args = argparse.ArgumentParser()
_args.add_argument("--rand",
                   type=str,
                   default="g",
                   help="random type (g|u) gaussian or uniform")

args = _args.parse_args()

# import matplotlib.pyplot as plt

# ## Define OutConv Functions
#
# $$
# \def\oconv{\circledast}
# $$
#
# $h$ is one-dimensional array
#
# outconv_211:
# $$g \oconv h h = \sum_{i,j} g(i,j) h(k - i) h(l - j)$$
#
# outconv_n11:
# $$g \oconv \vec{h}$$

def compute_sig_acc(X):
    s = np.linalg.svd(X, False, False)
    acc = np.zeros_like(s)
    acc[0] = s[0]
    for i in range(1, s.shape[0], 1):
        acc[i] = acc[i - 1] + s[i]
    return s, acc / np.sum(s)


def singular_value(X: np.ndarray, ndim=-1):
    if X.ndim > 2 and ndim != -1:
        X = np.reshape(X, [np.prod(X.shape[0:ndim]), -1])
    return np.linalg.svd(X, False, False)


suggest_rank = [1, 2, 4, 7, 9, 11, 15]

# filter
if args.rand == 'g':
    s_rank_matrix = lambda s, r: tensordec.randn_rank_matrix(s, r, True)
    s_rtensor = tensordec.randn_rtensor
    s_tucker = tensordec.randn_tucker
elif args.rand == 'u':
    s_rank_matrix = lambda s, r: tensordec.rand_rank_matrix(s, r, True)
    s_rtensor = tensordec.rand_rtensor
    s_tucker = tensordec.rand_tucker
else:
    raise ValueError("unknow args.rand %s" % args.rand)

# ## Check flip

# In[4]:

g = s_rank_matrix(8, 8)
h1 = np.random.rand(8)
h2 = np.random.rand(8)

lhs = outerconv(g, h1, h2)
rhs = outerconv(np.flip(g, (0, 1)), np.flip(h1), np.flip(h2))
rhs = np.flip(rhs, (0, 1))

record_append(
    "oconv-check-flip",
    r"check $g \oconv h_1 h_2$ = $\overline{\bar{g} \oconv \bar{h}_1 \bar{h}_2}$",
    diff=np.linalg.norm(lhs - rhs),
)

# ### Random matrix, check rank
#
# Rank of $g \circledast h_1 h_2$ equals to the rank of $g$.

task = []
for r in [1, 2, 3, 4, 6]:
    g = s_rank_matrix(9, r)
    hx = [s_rtensor(9) for _ in range(2)]
    out = outerconv(g, hx[0], hx[1])
    # print(f"g rank {r}, g shape {g.shape}, out shape {out.shape}")
    task.append({
        "kernel_rank": r,
        "kernel_shape": g.shape,
        "signal_length": hx[0].shape[-1],
        "singular_value": singular_value(out),
    })

record_append(
    "oconv-rank-after-outer-convolution",
    r"Rank of $g \circledast h_1 h_2$ equals to the rank of $g$",
    tasks=task,
)

# ### Conv zero
task = []
gs = 9
for r, rh1, rh2 in [(3, 1, 1), (3, 3, 2), (3, 3, 5), (7, 5, 3), (7, 5, 5),
                    (7, 5, 9), (7, 7, 7)]:
    g = s_rank_matrix(gs, r)
    hx = [
        zero_conv_signal_1d(np.ones(rh1 + 1), gs * 3),
        zero_conv_signal_1d(np.ones(rh2 + 1), gs * 3),
    ]
    # print(hx)
    # remove the value related to zero padding
    out = outerconv(g, hx[0], hx[1])[gs:-gs, gs:-gs]
    # print(f"g rank {r}, g shape {g.shape}, out shape {out.shape}")
    task.append({
        "kernel_rank": r,
        "kernel_shape": g.shape,
        "signal_init_len1": rh1,
        "signal_init_len2": rh2,
        "signal_length": hx[0].shape[-1],
        "singular_value": singular_value(out),
    })

record_append(
    "oconv-rank-zero-convolution",
    r"Rank of $g \circledast h_1 h_2$ equals to the rank of $g$",
    tasks=task,
)

# ### conv rank k kernel and rank r image

task = []
for i, j in [(1, 1), (1, 2), (2, 2), (3, 1), (2, 3), (3, 4)]:
    ker_rank = i
    img_rank = j
    ker_size = 7
    img_size = 32

    Ta = torch.matmul(torch.randn(img_size, img_rank),
                      torch.randn(img_rank, img_size))
    Ta = torch.reshape(Ta, (1, 1, img_size, img_size))
    Ta = Ta / torch.norm(Ta)
    Tk = torch.matmul(torch.randn(ker_size, ker_rank),
                      torch.randn(ker_rank, ker_size))
    Tk = torch.reshape(Tk, (1, 1, ker_size, ker_size))

    To = torch.conv2d(Ta, Tk)

    out = To[0, 0].data.cpu().numpy()

    task.append({
        "kernel_size": ker_size,
        "kernel_rank": ker_rank,
        "image_size": img_size,
        "image_rank": img_rank,
        "singular_value": singular_value(out),
    })

record_append(
    "oconv-rank-conv-kernel-image",
    r"rank $k$ kernel convolute rank $r$ image",
    tasks=task,
)

# ### many 1D h, Tucker rank

# G rank (2,3,4)
# h1 (5); h2 (5,5) rank (2,); h3 (5,5,5) rank (2,3,4)
shape = [3, 3, 3, 3]
rank = [2, 3, 3, 2]
g = s_rtensor(shape, rank)
hx = [
    s_rtensor(5),  # 1D
    s_rtensor(5),  # 1D
    s_rtensor([7, 7], 2),  # 2D, rank 2
    s_rtensor([9, 9, 18], [2, 3, 4]),  # 2D, rank 3
]

out = outerconv(g, hx)
G, Ax, sig = tensordec.hosvd(out, truncat=False, sig=True)

record_append(
    "oconv-outer-convolution-tucker-rank",
    r"outer convolution $g \oconv \vec{h}$ should have the same Tucker rank as $g$, where $h_i$ are 1D signals",
    kernel_rank=rank,
    kernel_shape=shape,
    signal_length=[h.shape for h in hx],
    signal_rank=[1, 1, [2, 2], [2, 3, 4]],
    reconstruct_error=np.linalg.norm(tensordec.tucker_build(G, Ax) - out),
    singular_values=sig,
)

# ### nD convolution

# In[10]:

task = []

ker_rank = [2, 4, 3]
ker_size = [6, 6, 6]
img_rank = [3, 2, 4]
img_size = [28, 28, 28]

G = s_tucker([1, 1, *img_size], [1, 1, *img_rank])
G = G / np.linalg.norm(G)
H = s_tucker([1, 1, *ker_size], [1, 1, *ker_rank])

out = torch.conv3d(torch.Tensor(G), torch.Tensor(H)).data.cpu().numpy()

ker, mat, sig = tensordec.hosvd(out, truncat=False, sig=True)

record_append(
    "oconv-rank-3D-conv",
    r"rank of 3D convolution equals tucker rank of kernel times tucker rank of image",
    kernel_size=ker_size,
    kernel_rank=ker_rank,
    image_size=img_size,
    image_rank=img_rank,
    reconstruct_error=np.linalg.norm(tensordec.tucker_build(ker, mat) - out),
    singular_values=sig,
)

# ## Check Volterra
#
# prepare data

# In[14]:

# ### check 9: volterra 2 & 2
#
# $$
# \begin{aligned}
#     y &= h_0 + h_1 * x + h_2 * x^2\\
#     z &= g_0 + g_1 * y + g_2 * y^2\\
# \end{aligned}
# $$
#
# Transform
#
# $$
# z \approx f_0 + f_1 * x + f_2 * x^2 + f_3 * x^3
# $$
#
# where
#
# $$
# \begin{aligned}
#     f_0 &= g_0 + h_0 \sum g_1 + h_0^2 \sum g_2 \\
#     f_1 &= g_1 \oconv h_1 + \sum_{\# h_0} g_2 \oconv h_0 h_1 + \sum_{\# h_0} g_2 \oconv h_1 h_0 \\
#     f_2 &= g_1 \oconv h_2 + g_2 \oconv h_1 h_1 + \sum_{\# h_0} g_2 \oconv h_0 h_2 + \sum_{\# h_0} g_2 \oconv h_2 h_0 \\
#     f_3 &= g_2 \oconv h_1 h_2 + g_2 \oconv h_2 h_1 \\
#     f_4 &= g_2 \oconv h_2 h_2 \\
# \end{aligned}
# $$

# In[29]:

h0, h1, h2 = np.random.randn(1), s_rtensor(5), s_rtensor([5, 5])
g0, g1, g2 = np.random.randn(1), s_rtensor(5), s_rtensor([5, 5])

x = s_rtensor(64)

y = h0 + conv_ordern(h1, x) + conv_ordern(h2, x)
z = g0 + conv_ordern(g1, y) + conv_ordern(g2, y)

f0 = g0 + g1.sum() * h0 + g2.sum() * h0 * h0
f1 = outerconv(g1, h1) + h0*outerconv(g2.sum(0), h1) + \
    h0*outerconv(g2.sum(1), h1)
f2 = outerconv(g1, h2) + outerconv(g2, h1, h1) + h0 * \
    outerconv(g2.sum(1), h2) + h0*outerconv(g2.sum(0), h2)
f3 = outerconv(g2, h1, h2) + outerconv(g2, h2, h1)
f4 = outerconv(g2, h2, h2)

pz = f0 + conv_ordern(f1, x) + conv_ordern(f2, x) + \
    conv_ordern(f3, x) + conv_ordern(f4, x)

record_append(
    "oconv-volterra-22",
    r"""Check order 2 Volterra - Order 2 Volterra -> Order 4 Volterra
\begin{equation}
\begin{aligned}
    y &= h_0 + h_1 * x + h_2 * x^2\\
    z &= g_0 + g_1 * y + g_2 * y^2\\
\end{aligned}
\end{equation}
will generate
\begin{equation}
    z = f_0 + f_1 * x + f_2 * x^2 + f_3 * x^3 + f_4 * x^4,
\end{equation}
where
\begin{aligned}
    f_0 &= g_0 + h_0 \sum g_1 + h_0^2 \sum g_2 \\
    f_1 &= g_1 \oconv h_1 + \sum_{\# h_0} g_2 \oconv h_0 h_1 + \sum_{\# h_0} g_2 \oconv h_1 h_0 \\
    f_2 &= g_1 \oconv h_2 + g_2 \oconv h_1 h_1 + \sum_{\# h_0} g_2 \oconv h_0 h_2 + \sum_{\# h_0} g_2 \oconv h_2 h_0 \\
    f_3 &= g_2 \oconv h_1 h_2 + g_2 \oconv h_2 h_1 \\
    f_4 &= g_2 \oconv h_2 h_2 \\
\end{aligned}""",
    h0_shape=h0.shape,
    h1_shape=h1.shape,
    h2_shape=h2.shape,
    g0_shape=g0.shape,
    g1_shape=g1.shape,
    g2_shape=g2.shape,
    x_shape=x.shape,
    ground_truth=z,
    estimate=pz,
    reconstruct_error=np.linalg.norm(z - pz),
)

# ## Check 2-layer Convolution Neuron Network
#
# using sigmoid activation
#
# $$
# \begin{aligned}
#     y &= \sigma(h * x)\\
#     z &= g * y\\
# \end{aligned}
# $$
# where $\sigma(x)$ is the sigmoid activation
# $$
# \sigma(x) = \dfrac{1}{1 + e^{-x}}.
# $$
#
# The Taylor expansion of sigmoid is
# $$
# \sigma(x) \approx
# \dfrac{1}{2} + \dfrac{x}{4} - \dfrac{x^3}{48} + \dfrac{x^5}{480} - \dfrac{17 x^7}{80640} + \dfrac{31 x^9}{1451520} - \dfrac{691 x^{11}}{319334400} + O(x^{13})
# $$

# In[30]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[31]:

x = s_rtensor(64)
h = s_rtensor(9)
g = s_rtensor(5)

y = sigmoid(conv_ordern(h, x))  # <-- we normalize here
z = conv_ordern(g, y)

pz = 0.5 * g.sum() \
    + 1.0/4 * conv_ordern(outerconv_diag(g, h), x)\
    - 1.0/48 * conv_ordern(outerconv_diag(g, [h, h, h]), x) \
    + 1.0/480 * conv_ordern(outerconv_diag(g, [h, h, h, h, h]), x)# \
# - 17.0/80640 * conv_ordern(outerconv_diag(g, [h, h, h, h, h, h, h]), x)

record_append(
    "oconv-conv-taylor",
    # r"""g * \sigma(h * x) \approx \dfrac{1}{2} + \dfrac{1}{4} (g \oconv h) * x - \dfrac{(\diag(g) \oconv h^3) * x^3}{48} + \dfrac{(\diag(g) \oconv h^5) * x^5}{480} - \dfrac{17 (\diag(g) \oconv h^7) * x^7}{80640} + O(x^{13})""",
    r"""g * \sigma(h * x) \approx \dfrac{1}{2} + \dfrac{1}{4} (g \oconv h) * x - \dfrac{(\diag(g) \oconv h^3) * x^3}{48} + \dfrac{(\diag(g) \oconv h^5) * x^5}{480} + O(x^{6})""",
    activation="sigmoid",
    x_shape=x.shape,
    h_shape=h.shape,
    g_shape=g.shape,
    ground_truth=z,
    estimate=pz,
    reconstruct_error=np.linalg.norm(z - pz),
)

# check approximate order-one proxy kernel
# Two-layer-network
# g * sigmoid(h * x)
# approximate
# - order-zero term
# - order-one term

g, h = s_rtensor(9), s_rtensor(9)

Tg = torch.FloatTensor(np.flip(g).copy().reshape(1, 1, -1))
Th = torch.FloatTensor(np.flip(h).copy().reshape(1, 1, -1))

model = HackerO1Net1D(g.shape[0] + h.shape[0] - 1)

model.fit(lambda x: torch.conv1d(torch.sigmoid(torch.conv1d(x, Th)), Tg),
          verbose=True)

comp_w0 = 0.5 * g.sum()
estm_w0 = model.order0

comp_w1 = 1.0 / 4 * outerconv_diag(g, h)
estm_w1 = model.order1

# check outerconv and convtranspose
_comp_w1 = 1 / 4 * torch.conv_transpose1d(Tg, Th, padding=0)
_comp_w1 = _comp_w1.squeeze().data.cpu().detach().numpy()
_comp_w1 = np.flip(_comp_w1).copy()

assert np.linalg.norm(
    _comp_w1 - comp_w1
) < 1e-7, f"compute error {np.linalg.norm(_comp_w1 - comp_w1)}\n {_comp_w1}\n {comp_w1}"

record_append(
    "oconv-conv-approximate-two-layer-net",
    r"""g * sigmoid(h * x) <-> A * x + b""",
    activation="sigmoid",
    num_layer="2",
    h=h,
    g=g,
    computed_w0=comp_w0,
    estimated_w0=estm_w0,
    computed_w1=comp_w1,
    estimated_w1=estm_w1,
    err_w0=np.linalg.norm(comp_w0 - estm_w0),
    err_w1=np.linalg.norm(comp_w1 - estm_w1),
)

# three layer network
# y * sigmoid(g * sigmoid(h * x))

f, g, h = s_rtensor(9), s_rtensor(9), s_rtensor(9)

Th = torch.FloatTensor(np.flip(h).copy().reshape(1, 1, -1))
Tg = torch.FloatTensor(np.flip(g).copy().reshape(1, 1, -1))
Tf = torch.FloatTensor(np.flip(f).copy().reshape(1, 1, -1))

model = HackerO1Net1D(f.shape[0] + g.shape[0] + h.shape[0] - 2)


def __net(x):
    _tx = torch.sigmoid(torch.conv1d(x, Th))
    _tx = torch.sigmoid(torch.conv1d(_tx, Tg))
    return torch.conv1d(_tx, Tf)


model.fit(__net, verbose=True)

estm_w0 = model.order0
estm_w1 = model.order1

sg = g.sum()
sf = f.sum()

comp_w0 = sf / 2 + sg / 8 * sf - sf / 48 * (sg / 2)**3 + sf / 480 * (sg / 2)**5

ocv_gh = 1 / 4 * outerconv(g, h)
alpha = 1 / 4 - 3 / 48 * (sg / 2)**2 + 5 / 480 * (sg / 2)**4
comp_w1 = alpha * outerconv(f, ocv_gh)

_comp_l1 = torch.conv_transpose1d(Th, Tg, padding=0)
_comp_l2 = torch.conv_transpose1d(_comp_l1, Tf, padding=0)
_comp_w1 = 1 / 4 * alpha * _comp_l2
_comp_w1 = _comp_w1.squeeze().data.cpu().detach().numpy()
_comp_w1 = np.flip(_comp_w1).copy()

assert np.linalg.norm(
    _comp_w1 - comp_w1
) < 1e-7, f"compute error {np.linalg.norm(_comp_w1 - comp_w1)}\n {_comp_w1}\n {comp_w1}"

record_append(
    "oconv-conv-approximate-three-layer-net",
    r"""f * sigmoid(g * sigmoid(h * x)) <-> A * x + b""",
    activation="sigmoid",
    num_layer="3",
    h=h,
    g=g,
    f=f,
    computed_w0=comp_w0,
    estimated_w0=estm_w0,
    computed_w1=comp_w1,
    estimated_w1=estm_w1,
    err_w0=np.linalg.norm(comp_w0 - estm_w0),
    err_w1=np.linalg.norm(comp_w1 - estm_w1),
)

# write to log file
record_dump_to_file(f"oconv-rank-result-{args.rand}")
# record_dump_to_file(stdout)
