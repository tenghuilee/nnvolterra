#%%
import matplotlib.pyplot as plt
import numpy as np

import npxconv
from plot_utils import mymarkers, savefig


#%%
# order n, perturbutation, statistic
def get_Hn(n: int):
    assert n >= 1, "n must >= 1"
    hshape = [5]*n
    Hn = np.random.randn(*hshape)
    Hn /= np.linalg.norm(Hn)

    return Hn
#%%
points = 32

# x = np.sin(np.linspace(np.pi, 4*np.pi, points))
x = np.random.randn(points)
x /= np.linalg.norm(x)
# noise = np.random.randn(points)
# noise /= np.linalg.norm(noise)
# noise *= 3.0
for eng in [0.5, 3]:
    y = x.copy()
    y[points//2] += eng

    # print(np.max(noise))

    # for k in range (1, 7):
    #     print(f"|y|_{2*k}^{k} = {np.linalg.norm(noise, 2*k)**k}")

    fig = plt.figure(figsize=(16,2))
    plt.plot(x, linestyle="-.")
    plt.plot(y)
    plt.legend(["$\mathbf{x}$", "$\mathbf{x} + \epsilon$"], loc="lower right")
    savefig(fig, f"perturbation-random-x-eps-{eng}")
    plt.close()

    print("start statistic")

    DiffHist = []
    for n in range(1, 9):
        dn = []
        print("\rcomputing n =", n, end="")
        for _ in range(1000):
            Hn = get_Hn(n)
            ox = npxconv.conv_ordern(Hn, [x]*n)
            oy = npxconv.conv_ordern(Hn, [y]*n)
            dn.append(np.linalg.norm(ox - oy))
        DiffHist.append(np.asarray(dn))
    print("  done")
    
    fig = plt.figure()
    plt.boxplot(DiffHist)
    savefig(fig, f"perturbation-boxplot-l2-norm-eps-{eng}")
    plt.close()

# order 5
def epsilon_order(Hn, noisy=16, noisyeng=0.1, points=512):
    n = len(Hn)
    x = np.sin(np.linspace(0, 8*np.pi, points))

    fig = plt.figure(figsize=(16,4))

    Hcy = 0
    for i in range(1, n+1):
        Hcy = Hcy + npxconv.conv_ordern(Hn[i-1], [x]*i)

    ye = x.copy()
    nloc = np.logspace(0, np.log10(points-1), noisy)
    nloc = nloc.astype(np.int32)
    # noisyG = noisyeng * np.sign(np.random.randn(points))
    # ye[nloc] += noisyG[nloc]
    ye[nloc] += noisyeng

    Hcye = 0

    for i in range(1, n+1):
        Hcye = Hcye + npxconv.conv_ordern(Hn[i-1], [ye]*i)
    
    plt.subplot(2,1,1)
    plt.plot(x, linestyle="-.")
    plt.plot(ye)
    plt.legend(["$\mathbf{x}$", "$\mathbf{x} + \epsilon$"], loc="lower right")
    plt.subplot(2,1,2)
    plt.plot(Hcy, linestyle="-.")
    plt.plot(Hcye)
    plt.legend(["$f(\mathbf{x})$", "$f(\mathbf{x} + \epsilon)$"], loc="lower right")

    tag = f"perturbation-order-{n}-noisy-{noisy}-eng-{noisyeng}"
    savefig(fig, tag)
    plt.close()
    print(tag, "norm diff output", np.linalg.norm(Hcye - Hcy) / np.linalg.norm(Hcy))
    print(tag, "norm diff input ", np.linalg.norm(ye - x) / np.linalg.norm(x))

# epsilon_order(11)
Hn = []
for i in range(1, 8+1):
    hshape = [5] * i
    H = np.random.randn(*hshape)
    H /= np.linalg.norm(H)
    Hn.append(H)

epsilon_order(Hn, noisy=30, noisyeng=0.1, points=500)
epsilon_order(Hn, noisy=30, noisyeng=0.2, points=500)

O1 = np.ones([8,])
O2 = np.ones([8,8])
O3 = np.ones([8,8,8])

goH = npxconv.outerconv(O1, O2)

fig = plt.figure()
plt.imshow(goH, cmap="copper")
# plt.colorbar()
plt.axis("off")
savefig(fig, "oconv-1d-2d")
plt.close()

gohh = npxconv.outerconv(O2, [O1, O1])

fig = plt.figure()
plt.imshow(gohh, cmap="copper")
# plt.colorbar()
plt.axis("off")
savefig(fig, "oconv-2d-1d-1d")
# plt.show()
plt.close()


gohh = npxconv.outerconv(O2, [O2, O2])

gos = gohh.shape
print(gos)
gohh = np.reshape(gohh, (gos[0] * gos[1], gos[2] * gos[3]))
print(gohh.shape)
fig = plt.figure()
plt.imshow(gohh, cmap="copper")
# plt.colorbar()
plt.axis("off")
savefig(fig, "oconv-2d-2d-2d")
# plt.show()
plt.close()

