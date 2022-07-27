"""
Compute the coefficient and order of stacking vconv layers
"""
import matplotlib.pyplot as plt

from plot_utils import mymarkers, savefig
from multinomial_utils import *

# incer = DigitInc(3, 4)
# while not incer.overflow():
#     print(incer, incer.digit_sum())
#     incer.inc()

print(multinomial_factors_sum(2, 2))
print("obj: 3 3 4 2 1")

# print(multinomial_factors_sum(5, 5))

# exit(0)

# nm_list = [(4, 4), (2, 4), (4, 2), (2, 3), (3, 2), (2, 2)]
nm_list = [
    (3, 4),
    (2, 4),
    (3, 3),
    (2, 3),
    (3, 2),
    (2, 2),
]

v_result = [multinomial_factors_sum(n, m) for n, m in nm_list]

fig = plt.figure(figsize=(12, 4))
# fig = plt.figure()
# plt.yscale("log")

for v, m in zip(v_result, mymarkers):
    print(v)
    plt.plot(v, marker=m)
fig.legend([f"n={n},m={m}" for n, m in nm_list])
plt.xlabel("order")
plt.ylabel("number of terms")
savefig(fig, "conv-multinomial")
# plt.show()
plt.close(fig)

# multinomial factors sum (5, 5)
# with open("multinomial_factor_sum_552.md", "w") as w:
#     w.write("| index |                    operation                    |\n")
#     w.write("| :---: | :---------------------------------------------: |\n")

#     multinomial_factors_sum_tex_print(5, 5, 2, file=w)
#     w.write("\n")

# for i in range(3, 15):
#     print(multinomial_factors_sum(i, 5))