import unittest
import numpy as np
import itertools

import libnpxconv as libxlib
import npxconv as xlib

class TestNpXConv(unittest.TestCase):
    # def test_main(self):
    #     a, b, c = np.random.randn(3, 9)
    #     libxlib.test(a, b, c)

    #     print(a)
    #     print(b)
    #     print(c)
    #     self.assertLessEqual(np.linalg.norm(c - a - b), 1e-8)

    @staticmethod
    def conv_o1(src: np.ndarray, ker: np.ndarray, stride: int = 1):
        olen = 1 + (src.shape[0] - ker.shape[0])//stride
        out = np.zeros(olen, dtype=src.dtype)

        for t in range(0, olen, 1):
            _y = 0.0
            for i in range(0, ker.shape[0], 1):
                _y += src[t*stride+i] * ker[i]
            out[t] = _y
        return out

    @staticmethod
    def outconv_211(g, h1, h2):
        """
        - g: shape (I1, I2)
        - h1: shape (I3)
        - h2: shape (I4)
        Return:
        - ans: shape (I1 + I3 - 1, I2 + I4 - 1)
        """
        out = np.zeros((g.shape[0] + h1.shape[0] - 1,
                        g.shape[1] + h2.shape[0] - 1), dtype=np.float64)

        for o1, o2 in itertools.product(range(out.shape[0]), range(out.shape[1])):
            _t = 0.0
            for i1, i2 in itertools.product(range(g.shape[0]), range(g.shape[1])):
                i3 = o1 - i1
                i4 = o2 - i2

                if i3 < 0 or i3 >= h1.shape[0]:
                    continue
                if i4 < 0 or i4 >= h2.shape[0]:
                    continue

                _t += g[i1, i2] * h1[i3] * h2[i4]
            out[o1, o2] = _t
        return out

    @staticmethod
    def outconv_n11(g, hn):
        """
        - g: shape (I1, I2 ... In)
        - hn: a list of(shape (J1, ... Jn)]
        Return:
        - ans: shape gshape + hnshape - 1
        """
        oshape = list(g.shape)
        for i, h in enumerate(hn):
            oshape[i] += h.shape[0] - 1

        out = np.zeros(oshape, dtype=np.float64)

        for ox in itertools.product(*[range(o) for o in oshape]):
            _t = 0.0
            for jx in itertools.product(*[range(s) for s in g.shape]):
                _hs = []
                for h, oi, ji in zip(hn, ox, jx):
                    _j = oi - ji
                    if _j < 0 or _j >= h.shape[0]:
                        _hs = None
                        break
                    _hs.append(h[_j])
                if _hs is None:
                    continue

                _t += g[tuple(jx)] * np.product(_hs)
            out[tuple(ox)] = _t
        return out

    # ofiles = open("222.log", "w")
    @staticmethod
    def outconv_222(g, h1, h2):
        Kg = np.max(g.shape)
        Kh = np.max(h1.shape)
        K2 = Kg + Kh - 1
        a = np.zeros([K2]*4)
        for i1, i2, j1, j2 in itertools.product(*[range(K2)]*4):
            _t = 0.0
            for k, l in itertools.product(range(g.shape[0]), range(g.shape[1])):
                if i1 - k < 0 or i1 - k >= h1.shape[0] or i2 - k < 0 or i2 - k >= h1.shape[1]:
                    continue
                if j1 - l < 0 or j1 - l >= h2.shape[0] or j2 - l < 0 or j2 - l >= h2.shape[1]:
                    continue
                _t += g[k, l] * h1[i1 - k, i2 - k] * h2[j1 - l, j2 - l]
                # print(f"[222] at {i1} {i2} {j1} {j2} ker {g[k,l]}, xt {h1[i1 - k, i2 - k] * h2[j1 - l, j2 - l]}", file=TestNpXConv.ofiles)
            a[i1, i2, j1, j2] = _t
        return a

    def test_o1(self):
        for stride in range(1, 5):
            klen = 9
            src = np.random.randn(32)
            ker = np.random.randn(klen)
            olen = 1 + (src.shape[0] - ker.shape[0]) // stride
            out1 = np.random.randn(olen)

            libxlib.conv1d_order_n(out1, ker, src)

            libxlib.conv1d_order_n(out1, ker, src, 0, stride)
            out2 = self.conv_o1(src, ker, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

    def test_conv_sparse(self):
        klen = 128
        for stride in range(1, 5):
            src = np.random.randn(512)
            ker = np.random.randn(klen)
            ker[np.abs(ker) < 0.5] = 0.0
            olen = 1 + (src.shape[0] - ker.shape[0]) // stride
            out1 = np.random.randn(olen)

            libxlib.conv1d_order_n(out1, ker, src)

            libxlib.conv1d_order_n(out1, ker, src, 0, stride)
            out2 = self.conv_o1(src, ker, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

    def test_oconv(self):
        g = np.random.randn(9, 9)
        h1 = np.random.randn(15)
        h2 = np.random.randn(15)

        out1 = self.outconv_211(g, h1, h2)

        out2 = np.empty_like(out1)

        libxlib.outerconv_211(out2, g, [h1, h2])
        self.assertLessEqual(np.linalg.norm(out2 - out1), 1e-8)

        libxlib.outerconv_nm(out2, g, [h1, h2])
        self.assertLessEqual(np.linalg.norm(out2 - out1), 1e-8)

    def test_oconv_222(self):
        g = np.random.randn(5, 5)
        h1 = np.random.randn(15, 9)
        h2 = np.random.randn(25, 7)

        out1 = self.outconv_222(g, h1, h2)
        out2 = np.empty_like(out1)

        libxlib.outerconv_nm(out2, g, [h1, h2])

        self.assertLessEqual(np.linalg.norm(out2 - out1), 1e-8)
    
    def test_oconv_diag(self):
        fg = np.random.randn(5)
        h1 = np.random.randn(15, 15)
        h2 = np.random.randn(25, 25)

        oshape = [
            h1.shape[0] + fg.shape[0] - 1,
            h1.shape[1] + fg.shape[0] - 1,
            h2.shape[0] + fg.shape[0] - 1,
            h2.shape[1] + fg.shape[0] - 1,
        ]

        out1 = np.zeros(oshape)
        out2 = np.empty_like(out1)

        libxlib.outerconv_nm(out1, np.diag(fg), [h1, h2])
        libxlib.outerconv_diagonal_nm(out2, fg, [h1, h2])
        self.assertLessEqual(np.linalg.norm(out2 - out1), 1e-8)

        libxlib.outerconv_nm(out1, np.diag(fg), [h1, h2], 0, 2)
        libxlib.outerconv_diagonal_nm(out2, fg, [h1, h2], 0, 2)
        self.assertLessEqual(np.linalg.norm(out2 - out1), 1e-8)

        with self.assertRaises((AssertionError, SystemError)):
            libxlib.outerconv_diagonal_nm(out2, fg, h1)
    
    def test_extend(self):
        for stride in range(1, 5):
            klen = 9
            src = np.random.randn(32)
            ker = np.random.randn(klen)
            olen = 1 + (src.shape[0] - ker.shape[0]) // stride
            out1 = np.random.randn(olen)
            out2 = np.zeros_like(out1)

            # order 1
            libxlib.conv1d_order_n(out1, ker, src, 8, stride)
            libxlib.conv1d_extend(out2, ker, src, 8, stride)

            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

            # order 2
            ker = np.random.randn(klen, klen)
            libxlib.conv1d_order_n(out1, ker, src, 8, stride)
            libxlib.conv1d_extend(out2, ker, src, 8, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

            # order 3
            ker = np.random.randn(klen, klen, klen)
            libxlib.conv1d_order_n(out1, ker, src, 8, stride)
            libxlib.conv1d_extend(out2, ker,  src, 8, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

    def test_nchn(self):
        for stride in range(1, 5):
            klen = 9
            src = np.random.randn(32)
            ker = np.random.randn(klen)
            olen = 1 + (src.shape[0] - ker.shape[0]) // stride

            # test n src and 1 src
            ker = np.random.randn(klen, klen)
            out1 = np.random.randn(olen)
            libxlib.conv1d_order_n(out1, ker, [src, src], 0, stride)
            out2 = np.random.randn(olen)
            libxlib.conv1d_order_n(out2, ker, src, 0, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

            ker = np.random.randn(klen, klen, klen)
            out1 = np.random.randn(olen)
            libxlib.conv1d_order_n(
                out1, ker, [src, src, src], 0, stride)
            out2 = np.random.randn(olen)
            libxlib.conv1d_order_n(out2, ker, src, 0, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

            ker = np.random.randn(klen, klen, klen, klen)
            out1 = np.random.randn(olen)
            libxlib.conv1d_order_n(out1, ker, [src, src, src, src],
                                   0, stride)
            out2 = np.random.randn(olen)
            libxlib.conv1d_order_n(out2, ker, src, 0, stride)
            self.assertLessEqual(np.linalg.norm(out1 - out2), 1e-8)

    def test_conv_o1(self):
        klen = 9
        src = np.random.randn(32)
        ker = np.random.randn(klen)
        olen = src.shape[0] + ker.shape[0] - 1
        out1 = np.random.randn(olen)

        out1 = self.conv_o1(np.pad(src, ((klen-1, klen-1))), ker)
        out2 = np.zeros_like(out1)

        libxlib.conv1d_order_n(out2, ker, src, klen-1)

        self.assertLessEqual(np.linalg.norm(out2 - out1), 1e-8)

    def test_combinations(self):
        g = np.random.randn(5)
        h = np.random.randn(5)

        x = np.random.randn(63)

        y = xlib.conv_ordern(h, x)
        z = xlib.conv_ordern(g, y)

        gh = xlib.outerconv(g, h)

        ez = xlib.conv_ordern(gh, x)

        self.assertLessEqual(np.linalg.norm(ez - z), 1e-8)

    def test_oconv_stride(self):
        g = np.random.randn(7)
        h = np.random.randn(5)
        u = np.random.randn(9)

        x = np.random.randn(255)

        y = xlib.conv_ordern(h, x, stride=2)
        z = xlib.conv_ordern(g, y, stride=2)

        gh = xlib.outerconv(g, h, stride=2)
        ez = xlib.conv_ordern(gh, x, stride=4)
        self.assertLessEqual(np.linalg.norm(ez - z), 1e-8)

        y = xlib.conv_ordern(h, x, stride=2)
        z = xlib.conv_ordern(g, y, stride=3)

        gh = xlib.outerconv(g, h, stride=2)
        ez = xlib.conv_ordern(gh, x, stride=6)

        self.assertLessEqual(np.linalg.norm(ez - z), 1e-8)
    
        z = xlib.conv_ordern(h, x, stride=2)
        z = xlib.conv_ordern(g, z, stride=3)
        z = xlib.conv_ordern(u, z, stride=4)

        gh = xlib.outerconv(u, xlib.outerconv(g, h, stride=2), stride=6)
        ez = xlib.conv_ordern(gh, x, stride=24)

        self.assertLessEqual(np.linalg.norm(ez - z), 1e-8)
