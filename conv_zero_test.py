import unittest
import numpy as np
import scipy.signal

from conv_zero import *


class ConvZeroCheck(unittest.TestCase):
    def test_conv1d(self):
        kernel = np.random.randn(5)
        olen = 32
        init = np.random.randn(5)
        ans = zero_conv_signal_1d(kernel, olen, init)

        cvres = np.convolve(ans, kernel, mode="valid")

        ncv = np.linalg.norm(cvres) / np.linalg.norm(ans)

        self.assertLessEqual(ncv, 1e-8)

    
    def test_conv2d(self):
        kernel = np.random.randn(3,3)
        osize = (32, 32)

        ans = zero_conv_signal_2d(kernel, osize)

        cvres = scipy.signal.convolve2d(ans, kernel, mode="valid")


        ncv = np.linalg.norm(cvres) / np.linalg.norm(ans)

        self.assertLessEqual(ncv, 1e-8)
