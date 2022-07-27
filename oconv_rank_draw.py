import argparse
import json
from tkinter.tix import MAX

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers

from record_utils import record_load_file
from plot_utils import mymarkers, savefig

_args = argparse.ArgumentParser()
_args.add_argument("--rand",
                   type=str,
                   default="g",
                   help="random type (g|u) gaussian or uniform")

args = _args.parse_args()

MIN_SINGULAR_VALUE = 1e-16
MAX_SINGULAR_VALUE = 1e16

JLogDict = record_load_file(f"oconv-rank-result-{args.rand}")

# volterra 22
record = JLogDict["oconv-volterra-22"]
print(record["info"])
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["ground_truth"], marker=mymarkers[0], linestyle='dashed')
# axi.set_ylabel('y')
# plt.show()
# exit(0)
savefig(fig, f"oconv-volterra-22-ground-truth-{args.rand}")
plt.close(fig)

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["estimate"], marker=mymarkers[1], linestyle='dashed')
# axi.set_ylabel('y')
savefig(fig, f"oconv-volterra-22-estimated-{args.rand}")
plt.close(fig)
# exit()

# conv taylor
record = JLogDict["oconv-conv-taylor"]
print(record["info"])
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["ground_truth"], marker=mymarkers[0], linestyle='dashed')
# axi.set_ylabel('y')
savefig(fig, f"oconv-conv-taylor-ground-truth-{args.rand}")
plt.close(fig)

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["estimate"], marker=mymarkers[1], linestyle='dashed')
# axi.set_ylabel('y')
savefig(fig, f"oconv-conv-taylor-estimated-{args.rand}")
plt.close(fig)

# rank_after_outer_convolution
record = JLogDict["oconv-rank-after-outer-convolution"]
print(record["info"])

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.set_yscale('log')
for mark, task in zip(mymarkers, record["tasks"]):
    ksize = task["signal_length"] // 2
    yy = task["singular_value"][0:-ksize]
    yy = np.clip(yy, MIN_SINGULAR_VALUE, MAX_SINGULAR_VALUE)
    axi.plot(yy, marker=mark, linestyle='dashed')
axi.legend(["$r_g = %d$" % t["kernel_rank"] for t in record["tasks"]],
           loc='lower right')
axi.set_ylabel("singular value ($log_{10}$)")
savefig(fig, f"oconv-rank-after-outer-convolution-{args.rand}")
plt.close(fig)
# fig.show()

# rank_3D_conv
record = JLogDict["oconv-rank-3D-conv"]
print(record["info"])
print(record["kernel_rank"], record["image_rank"])
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.set_yscale('log')
for mark, sig in zip(mymarkers, record["singular_values"][2::]):
    sig = np.clip(sig, MIN_SINGULAR_VALUE, MAX_SINGULAR_VALUE)
    axi.plot(sig, marker=mark, linestyle='dashed')

legend = []
count = len(record["kernel_rank"])
for i, rg, rh in zip(range(1, count + 1), record["kernel_rank"],
                     record["image_rank"]):
    legend.append(f"$r_{{g{i}}} r_{{h{i}}} = {rg} \\times {rh} = {rg * rh}$")

axi.legend(legend, loc='upper right')
axi.set_ylabel("singular value ($log_{10}$)")
savefig(fig, f"oconv-rank-3D-conv-{args.rand}")
plt.close(fig)
# plt.show()

# rank_conv_kernel_image
record = JLogDict["oconv-rank-conv-kernel-image"]
print(record["info"])

task = record["tasks"]
rankList = []
for i, t in enumerate(task):
    rankList.append((i, t["kernel_rank"] * t["image_rank"], t["kernel_rank"],
                     t["image_rank"]))

rankList.sort(key=lambda x: x[1])

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.set_yscale('log')
for mark, idx in zip(mymarkers, rankList):
    yy = task[idx[0]]["singular_value"]
    yy = np.clip(yy, MIN_SINGULAR_VALUE, MAX_SINGULAR_VALUE)
    axi.plot(yy, marker=mark, linestyle='dashed')

axi.legend(
    [fr"$r_g r_h = {rl[2]} \times {rl[3]} = {rl[1]}$" for rl in rankList],
    loc='best')
axi.set_ylabel("singular value ($log_{10}$)")
savefig(fig, f"oconv-rank-conv-kernel-image-{args.rand}")
plt.close(fig)
# fig.show()

# outer_convolution_tucker_rank
record = JLogDict["oconv-outer-convolution-tucker-rank"]
print(record["info"])

sig = record["singular_values"]
fig = plt.figure(figsize=(8,5)) # need higher
axi = fig.add_subplot(1, 1, 1)
axi.set_yscale('log')
for mark, s in zip(mymarkers, sig):
    s = np.clip(s, MIN_SINGULAR_VALUE, MAX_SINGULAR_VALUE)
    axi.plot(s, marker=mark, linestyle='dashed')
legend = []
for i, rg, sg, rh in zip(range(1,
                               len(sig) + 1), record["kernel_rank"],
                         record["kernel_shape"], record["signal_rank"]):
    if isinstance(rh, (list, tuple)):
        for r in rh:
            legend.append(
                f"$r_{{g{i}}} = {rg}, z_{i} r_{{h{i}}} = {sg} \\times {r} = {sg * r}$"
            )
    else:
        legend.append(
            f"$z_{i} = {sg}, r_{{g{i}}} r_{{h{i}}} = {rg} \\times {rh} = {rg * rh}$"
        )

axi.legend(legend, loc='lower right')
axi.set_ylabel("singular value ($log_{10}$)")
savefig(fig, f"oconv-outer-convolution-tucker-rank-{args.rand}")
# fig.show()

# rank_after_outer_convolution
record = JLogDict["oconv-rank-zero-convolution"]
print(record["info"])

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.set_yscale('log')
print("num tasks", len(record["tasks"]))
legend = []
for mark, task in zip(mymarkers, record["tasks"]):
    ksize = task["signal_length"] // 2
    yy = task["singular_value"]
    yy = np.clip(yy, MIN_SINGULAR_VALUE, MAX_SINGULAR_VALUE)
    axi.plot(yy, marker=mark, linestyle='dashed')
    rg = task['kernel_rank']
    rh1 = task["signal_init_len1"]
    rh2 = task["signal_init_len2"]
    # legend.append(
    #     f"$\min(r_g = {rg}, T_1 = {rh1}, T_2 = {rh2})={min(rg, rh1, rh2)}$")
    legend.append(
        f"$\min({rg}, {rh1}, {rh2})={min(rg, rh1, rh2)}$")

axi.legend(legend, loc='lower right')
axi.set_ylabel("singular value ($log_{10}$)")
savefig(fig, f"oconv-rank-zero-convolution-{args.rand}")
plt.close(fig)

# list
# - oconv-conv-approximate-two-layer-net
# - oconv-conv-approximate-three-layer-net

for l in ["two", "three"]:
    record = JLogDict[f"oconv-conv-approximate-{l}-layer-net"]
    print(record["info"])

    fig = plt.figure()
    axi = fig.add_subplot(1, 1, 1)
    axi.plot(record["computed_w1"], marker=mymarkers[0], linestyle="dashed")
    savefig(fig,
            f"oconv-conv-approximate-{l}-layer-net-w1-computed-{args.rand}")
    plt.close(fig)

    fig = plt.figure()
    axi = fig.add_subplot(1, 1, 1)
    axi.plot(record["estimated_w1"], marker=mymarkers[1], linestyle="dashed")
    savefig(fig,
            f"oconv-conv-approximate-{l}-layer-net-w1-estimated-{args.rand}")
    plt.close(fig)
