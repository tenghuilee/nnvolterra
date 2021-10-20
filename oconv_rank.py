#!/usr/bin/env python
# coding: utf-8

# # Rank for outer convolution And Check For Volterra

import time

import tensordec
import numpy as np
import torch
from npxconv import conv_ordern, outerconv, outerconv_diag
import json

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

def randn_rank_matrix(shape, rank, isnorm=True):
    if isinstance(shape, int):
        shape = (shape, shape)
    else:
        assert len(shape) == 2

    _L = np.random.randn(shape[0], rank)
    _R = np.random.randn(rank, shape[1])

    ans = np.matmul(_L, _R)
    if isnorm:
        ans /= np.linalg.norm(ans)
    return ans


def randn_tucker(shape, rank, isnorm=True):
    ans = tensordec.randn_tucker(shape, rank)
    if isnorm:
        ans /= np.linalg.norm(ans)
    return ans


def randn_rtensor(shape: list, rank=None, isnorm=True):
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    if rank is None:
        ans = np.random.randn(*shape)
        if isnorm:
            ans /= np.linalg.norm(ans)
        return ans

    if len(shape) == 2:
        return randn_rank_matrix(shape, rank, isnorm)
    else:
        return randn_tucker(shape, rank, isnorm)


def compute_sig_acc(X):
    s = np.linalg.svd(X, False, False)
    acc = np.zeros_like(s)
    acc[0] = s[0]
    for i in range(1, s.shape[0], 1):
        acc[i] = acc[i-1] + s[i]
    return s, acc / np.sum(s)


def singular_val(X: np.ndarray, ndim=-1):
    if X.ndim > 2 and ndim != -1:
        X = np.reshape(X, [np.prod(X.shape[0:ndim]), -1])
    return np.linalg.svd(X, False, False)


JLogDict = dict()
JLogDict["date"] = time.asctime(time.localtime())


def append_record(tag: str, info: str, **kwargs):
    print("append", tag)
    print("info", info)

    JLogDict[tag] = {
        "info": info,
        **kwargs,
    }


suggest_rank = [1, 2, 4, 7, 9, 11, 15]

# ## Check flip

# In[4]:


g = randn_rank_matrix(8, 8)
h1 = np.random.rand(8)
h2 = np.random.rand(8)

lhs = outerconv(g, h1, h2)
rhs = outerconv(np.flip(g, (0, 1)), np.flip(h1), np.flip(h2))
rhs = np.flip(rhs, (0, 1))

append_record(
    "oconv-check-flip",
    r"check $g \oconv h_1 h_2$ = $\overline{\bar{g} \oconv \bar{h}_1 \bar{h}_2}$",
    diff=np.linalg.norm(lhs - rhs),
)

# ### Random matrix, check rank
#
# Rank of $g \circledast h_1 h_2$ equals to the rank of $g$.

task = []
for r in [1, 2, 3, 4, 6]:
    g = randn_rank_matrix(9, r)
    hx = [randn_rtensor(9) for _ in range(2)]
    out = outerconv(g, hx[0], hx[1])
    # print(f"g rank {r}, shape {g.shape}")
    task.append({
        "kernel_rank": r,
        "kernel_shape": g.shape,
        "signal_length": hx[0].shape[-1],
        "singular_value": singular_val(out),
    })

append_record(
    "oconv-rank-after-outer-convolution",
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
        "singular_value": singular_val(out),
    })

append_record(
    "oconv-rank-conv-kernel-image",
    r"rank $k$ kernel convolute rank $r$ image",
    tasks=task,
)

# ### many 1D h, Tucker rank

# G rank (2,3,4)
# h1 (5); h2 (5,5) rank (2,); h3 (5,5,5) rank (2,3,4)
shape = [3, 3, 3, 3]
rank = [2, 3, 3, 2]
g = randn_rtensor(shape, rank)
hx = [
    randn_rtensor(5),  # 1D
    randn_rtensor(5),  # 1D
    randn_rtensor([7, 7], 2),  # 2D, rank 2
    randn_rtensor([9, 9, 13], [2, 3, 4]),  # 2D, rank 3
]

out = outerconv(g, hx)
G, Ax, sig = tensordec.hosvd(out, truncat=False, sig=True)

append_record(
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

ker_rank = [2, 3, 4]
ker_size = [5, 5, 5]
img_rank = [3, 4, 5]
img_size = [28, 28, 28]

G = tensordec.randn_tucker([1, 1, *img_size], [1, 1, *img_rank])
G = G / np.linalg.norm(G)
H = tensordec.randn_tucker([1, 1, *ker_size], [1, 1, *ker_rank])

out = torch.conv3d(torch.Tensor(G), torch.Tensor(H)).data.cpu().numpy()

ker, mat, sig = tensordec.hosvd(out, truncat=False, sig=True)

append_record(
    "oconv-rank-3D-conv",
    r"rank of 3D convolution equals tucker rank of kernel times tucker rank of image",
    kernel_size=ker_size,
    kernel_rank=ker_rank,
    image_size=img_size,
    image_rank=img_rank,
    reconstruct_error=np.linalg.norm(tensordec.tucker_build(ker, mat) - out),
    singular_values=sig,
)

# ## h is 2D
#
# If flattened, => h is 1D.

# In[12]:

task = []
for _ in range(8):
    ker_rank = np.random.randint(1, 5, (2,))
    ker_size = [
        np.random.randint(ker_rank[i], ker_rank[1]+9, (2,))
        for i in range(2)
    ]
    img_rank = np.random.randint(1, 5)
    img_size = np.random.randint(img_rank, img_rank+9, (2,))

    g = randn_rank_matrix(img_size, img_rank)
    h0 = randn_rank_matrix(ker_size[0], ker_rank[0])
    h1 = randn_rank_matrix(ker_size[1], ker_rank[1])

    out = outerconv(g, h0, h1)

    g, Ax, sig = tensordec.hosvd(out, truncat=True, sig=True)

    task.append({
        "kernel_size_0": ker_size[0],
        "kernel_size_1": ker_size[1],
        "kernel_rank_0": ker_rank[0],
        "kernel_rank_1": ker_rank[1],
        "img_size": img_size,
        "img_rank": img_rank,
        "reconstruct_error": np.linalg.norm(tensordec.tucker_build(g, Ax) - out),
        "singular_values": sig,
        "flatten_singular_values": singular_val(out, 2),
    })

append_record(
    "oconv-outer-convolution-h-2d",
    r"outer convolution $g \oconv h_1 h_2$ where $g$, $h_1$ and $h_2$ are both 2D signal",
    task=task,
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

h0, h1, h2 = np.random.randn(1), randn_rtensor(5), randn_rtensor([5, 5])
g0, g1, g2 = np.random.randn(1), randn_rtensor(5), randn_rtensor([5, 5])

x = randn_rtensor(64)

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

append_record(
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

x = randn_rtensor(64)
h = randn_rtensor(9)
g = randn_rtensor(5)

y = sigmoid(conv_ordern(h, x))  # <-- we normalize here
z = conv_ordern(g, y)

pz = 0.5 * g.sum() \
    + 1.0/4 * conv_ordern(outerconv_diag(g, h), x)\
    - 1.0/48 * conv_ordern(outerconv_diag(g, [h, h, h]), x) \
    + 1.0/480 * conv_ordern(outerconv_diag(g, [h, h, h, h, h]), x)# \
    # - 17.0/80640 * conv_ordern(outerconv_diag(g, [h, h, h, h, h, h, h]), x)

append_record(
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

# write to log file
class _jsdec(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


with open("result/rank for outer conv result.json", "w") as rec:
    json.dump(JLogDict, rec, cls=_jsdec, indent=2)

print("done")
