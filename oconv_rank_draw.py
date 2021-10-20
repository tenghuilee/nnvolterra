import numpy as np
import json

import matplotlib.pyplot as plt

# 16, 9
plt.rcParams['figure.figsize'] = 8, 4.5
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.handlelength'] = 3

mymarkers = [".", "^", "+", "x", "*", ",", "|", "d", "_", "v"]

with open("result/rank for outer conv result.json", "r") as rec:
    JLogDict = json.load(rec)


def save_fig(fig, tag):
    # fig.savefig(f"result/{tag}.eps", dpi=500, bbox_inches='tight', pad_inches=0)
    # fig.savefig(f"result/{tag}.jpg", dpi=500, bbox_inches='tight', pad_inches=0)
    # top, bottom | left, right in [0, 1]
    fig.subplots_adjust(top=0.99, bottom=0.06, right=0.99, left=0.1, hspace=0, wspace=0)
    # all values in [0,1]
    # fig.gca().margins(0.03, 0.03, x=None, y=None, tight=False)
    
    fig.savefig(f"result/{tag}.eps", dpi=600, bbox_inches=0, transparent=True, pad_inches=0)
    # fig.savefig(f"result/{tag}.pdf", dpi=600, bbox_inches=0, transparent=True, pad_inches=0)
    # fig.savefig(f"result/{tag}.png", dpi=600, bbox_inches=0, transparent=True, pad_inches=0)
    fig.savefig(f"result/{tag}.jpg", dpi=600, bbox_inches=0, transparent=True, pad_inches=0)

# rank_after_outer_convolution
record = JLogDict["oconv-rank-after-outer-convolution"]
print(record["info"])

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
for mark, task in zip(mymarkers, record["tasks"]):
    ksize = task["signal_length"] // 2
    axi.plot(task["singular_value"][0:-ksize], marker=mark, linestyle='dashed')
axi.legend(["$r_g = %d$" % t["kernel_rank"] for t in record["tasks"]])
axi.set_ylabel("singular value")
save_fig(fig, "oconv-rank-after-outer-convolution")
# fig.show()
# exit()

# volterra 22
record = JLogDict["oconv-volterra-22"]
print(record["info"])
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["ground_truth"], marker=mymarkers[0], linestyle='dashed')
axi.set_ylabel('y')
save_fig(fig, "oconv-volterra-22-ground-truth")

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["estimate"], marker=mymarkers[1], linestyle='dashed')
axi.set_ylabel('y')
save_fig(fig, "oconv-volterra-22-estimated")

# exit()

# conv taylor
record = JLogDict["oconv-conv-taylor"]
print(record["info"])
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["ground_truth"], marker=mymarkers[0], linestyle='dashed')
axi.set_ylabel('y')
save_fig(fig, "oconv-conv-taylor-ground-truth")

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
axi.plot(record["estimate"], marker=mymarkers[1], linestyle='dashed')
axi.set_ylabel('y')
save_fig(fig, "oconv-conv-taylor-estimated")

# rank_3D_conv
record = JLogDict["oconv-rank-3D-conv"]
print(record["info"])
print(record["kernel_rank"], record["image_rank"])
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
for mark, sig in zip(mymarkers, record["singular_values"][2::]):
    axi.plot(sig, marker=mark, linestyle='dashed')

axi.legend([fr"$r_k r_g = {rk} \times {rg} = {rk * rg}$" for rk,
            rg in zip(record["kernel_rank"], record["image_rank"])])
axi.set_ylabel("signular value")
save_fig(fig, "oconv-rank-3D-conv")
# plt.show()


# rank_conv_kernel_image
record = JLogDict["oconv-rank-conv-kernel-image"]
print(record["info"])

task = record["tasks"]
rankList = []
for i, t in enumerate(task):
    rankList.append((i, t["kernel_rank"] * t["image_rank"],
                     t["kernel_rank"], t["image_rank"]))

rankList.sort(key=lambda x: x[1])

fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
for mark, idx in zip(mymarkers, rankList):
    axi.plot(task[idx[0]]["singular_value"], marker=mark, linestyle='dashed')

axi.legend(
    [fr"$r_g r_h = {rl[2]} \times {rl[3]} = {rl[1]}$" for rl in rankList])
axi.set_ylabel("singular value")
save_fig(fig, "oconv-rank-conv-kernel-image")
# fig.show()

# outer_convolution_tucker_rank
record = JLogDict["oconv-outer-convolution-tucker-rank"]
print(record["info"])

sig = record["singular_values"]
fig = plt.figure()
axi = fig.add_subplot(1, 1, 1)
for mark, s in zip(mymarkers, sig):
    axi.plot(s, marker=mark, linestyle='dashed')
legend = []
for rg, sg, rh in zip(record["kernel_rank"], record["kernel_shape"], record["signal_rank"]):
    if isinstance(rh, (list, tuple)):
        for r in rh:
           legend.append(fr"$r_g = {rg}, ~ s_g r_h = {sg} \times {r} = {sg * r}$")
    else:
        legend.append(fr"$s_g = {sg}, ~ s_g r_h = {rg} \times {rh} = {rg * rh}$")
    
axi.legend(legend)
axi.set_ylabel("singular value")
save_fig(fig, "oconv-outer-convolution-tucker-rank")
# fig.show()
