import numpy as np
import itertools

# import scipy.signal
# def check_zero_conv(func):
#     def __inner(kernel, olen, init=None):
#         ans = func(kernel, olen, init)  # type: np.ndarray
#         if ans.ndim == 1:
#             conv = np.convolve(ans, kernel, mode="valid")
#         else:
#             conv = scipy.signal.convolve2d(ans, kernel, mode="valid")
#         ncv = np.linalg.norm(conv) / np.size(conv)
#         assert ncv < 1e-8, f"conv = {ncv} is not zero.\n{np.round(conv, 1)}"
#         return ans
#     return __inner


# @check_zero_conv
def zero_conv_signal_1d(kernel, olen, init=None):
    """
    sum_{i} kernel[i] signal[t + i]
    """
    kernel = np.flip(kernel)  # flipped
    klen = len(kernel)
    assert kernel[-1] != 0

    plen = klen - 1

    out = np.zeros(olen)

    kend = -1 / kernel[-1]

    if init is None:
        init = np.random.randn(plen)

    out[0:plen] = init[0:plen]

    for i in range(olen - klen + 1):
        x = kend * kernel[0:plen] @ out[i:i + plen]
        out[i + plen] = x

    return out


def ndrange(*args):
    ndlists = []
    for x in args:
        if isinstance(x, int):
            ndlists.append(range(x))
        elif isinstance(x, (tuple, list)):
            ndlists.append(range(*x))
        else:
            raise ValueError(f"not supported {x}")

    for p in itertools.product(*ndlists):
        yield p


# @check_zero_conv
def zero_conv_signal_2d(kernel, osize, init=None):
    """
    sum_{i,j} kernel[i, j] signal[t_1 + i, t_2 + j]
    """
    ksize = kernel.shape
    kernel = np.flip(kernel, axis=(0, 1))
    assert kernel[-1, -1] != 0

    out = np.zeros(osize)

    kend = -1 / kernel[-1, -1]
    if init is None:
        init = np.random.randn(*osize)

    out[0:ksize[0] - 1, :] = init[0:ksize[0] - 1, :]
    out[:, 0:ksize[1] - 1] = init[:, 0:ksize[1] - 1]

    for t1, t2 in ndrange(osize[0] - ksize[0] + 1, osize[1] - ksize[0] + 1):
        x = 0
        for i, j in ndrange(ksize[0], ksize[1]):
            if i == ksize[0] - 1 and j == ksize[1] - 1:
                continue
            x += kernel[i, j] * out[t1 + i, t2 + j]
        out[t1 + ksize[0] - 1, t2 + ksize[1] - 1] = x * kend

    return out
