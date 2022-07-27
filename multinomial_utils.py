"""
Compute the coefficient and order of stacking vconv layers
"""
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba.experimental import jitclass

from plot_utils import mymarkers, savefig

@jitclass([
    ('base', numba.int32),  # a simple scalar field
    ('num_digits', numba.int32),  # a simple scalar field
    ('digits', numba.int32[:]),  # an array field
])
class DigitIncBase:

    def __init__(self, base, num_digits) -> None:
        self.base = base
        # little endian
        self.digits = np.zeros(num_digits + 1, dtype=np.int32)
        self.num_digits = num_digits

    def inc(self):
        for i in range(self.num_digits + 1):
            self.digits[i] += 1
            if self.digits[i] >= self.base:
                self.digits[i] = 0
                # adding one is in next loop
            else:
                break
        return self.digits

    def overflow(self) -> bool:
        return self.digits[-1] != 0

    def digit_sum(self):
        return np.sum(self.digits[0:-1])


class DigitInc:

    def __init__(self, base, num_digits, brace="[]") -> None:
        self.incer = DigitIncBase(base, num_digits)
        self.brace = brace

    @property
    def digits(self):
        return self.incer.digits

    def inc(self):
        return self.incer.inc()

    def overflow(self) -> bool:
        return self.incer.overflow()

    def digit_sum(self):
        return self.incer.digit_sum()

    def __str__(self) -> str:
        digits = self.incer.digits
        ans = f"{self.brace[0]}{digits[-2]}"
        for d in reversed(digits[0:-2]):
            ans += f" {d}"
        ans += self.brace[1]
        return ans


# incer = DigitInc(3, 4)
# while not incer.overflow():
#     print(incer, incer.digit_sum())
#     incer.inc()


@numba.jit()
def multinomial_factors(n: int, m: int, out=None):
    """
    (x_0 + x_1 + ... + x_n)^m
    n,m > 0
    - n: the first order
    - m: the second order
    return: coefficients
    """
    incer = DigitIncBase(n, m)

    if out is None:
        out = np.zeros((n - 1) * m + 1, dtype=np.int64)

    while not incer.overflow():
        ds = incer.digit_sum()
        out[ds] += 1
        # print(incer, incer.digit_sum())
        incer.inc()

    return out


# for i in range(14):
#     ans = multinomial_factors(2, i)
#     print(str(ans)[1:-1].center(81))

# print(multinomial_factors(3, 2))


@numba.njit()
def multinomial_factors_sum(n: int, m: int):
    """
    (x_0 + x_1 + ... + x_n)^0 + 
    (x_0 + x_1 + ... + x_n)^1 + 
    ...
    (x_0 + x_1 + ... + x_n)^m + 

    n,m > 0
    - n: the first order
    - m: the second order
    return: coefficients
    """

    cff = np.zeros(n * m + 1, dtype=np.int64)
    n += 1  # start from zero
    m += 1  # start from zero

    for i in numba.prange(0, m):
        multinomial_factors(n, i, cff)

    return cff


def multinomial_factors_sum_tex_print(n: int, m: int, order: int, file=None):
    """
    nice print version

    (x_0 + x_1 + ... + x_n)^0 + 
    (x_0 + x_1 + ... + x_n)^1 + 
    ...
    (x_0 + x_1 + ... + x_n)^m + 

    n,m > 0
    - n: the first order
    - m: the second order
    return: coefficients
    """
    counter = 0
    # start from zero, m + 1, n + 1
    for i in range(n+1):
        incer = DigitInc(m+1, i)

        while not incer.overflow():
            if incer.digit_sum() == order:
                counter += 1
                # to string
                print("| %02d | " % counter, "$G_%d" % i, "\\circledast \\left\\lceil ", end="", file=file)
                for d in incer.digits[0:-2]:
                    print(f"H_{d}, ", end="", file=file)
                print(f"H_{incer.digits[-2]} \\right\\rfloor$ |", file=file)

            incer.inc()


# print(multinomial_factors_sum(2, 2))
# print("obj: 3 3 4 2 1")
