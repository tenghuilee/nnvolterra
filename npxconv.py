import libnpxconv as libx

import numpy as np

# params for libx (sedt, kernel, srcx, /, padleft, stride)

def conv_ordern(ker, *src, oshape=None, padleft=0, stride=1):
    if len(src) == 1:
        src = src[0]

    if oshape is None:
        if isinstance(src, (list, tuple)):
            assert np.ndim(ker) == len(src)
            oshape = np.min([np.min(s.shape) for s in src])
        else:
            oshape = np.min(src.shape)
        ks = np.max(ker.shape)
        if oshape > ks:
            oshape = (oshape - ks) // stride + 1

    out = np.zeros(oshape, dtype=ker.dtype)

    if isinstance(src, list):
        src = [np.ascontiguousarray(x) for x in src]
    else:
        src = np.ascontiguousarray(src)
    ker = np.ascontiguousarray(ker)
    libx.conv1d_order_n(out, ker, src, padleft, stride)
    return out


def outerconv(g, *hx, oshape=None, padleft=0, stride=1):
    if isinstance(hx[0], (list, tuple)) and len(hx) == 1:
        hx = hx[0]
    assert np.ndim(g) == len(hx), \
        f"mismatch ndim(g) = {np.ndim(g)}, len(hx) = {len(hx)}"
    if oshape is None:
        oshape = []
        for i, h in enumerate(hx):
            gsi = g.shape[i]
            for hsi in h.shape:
                oshape.append(hsi + stride*gsi-stride)

    out = np.zeros(oshape, dtype=g.dtype)
    g = np.ascontiguousarray(g)
    hx = [np.ascontiguousarray(h) for h in hx]

    libx.outerconv_nm(out, g, hx, padleft, stride)
    return out

def outerconv_diag(g, *hx, oshape=None, padleft=0, stride=1):
    # in the case of passing [x, y, z, ...]
    if isinstance(hx[0], (list, tuple)) and len(hx) == 1:
        hx = hx[0]
    assert np.ndim(g) == 1, "only 1D signal supported"
    assert len(hx) >= 1, "length of hx must grater than or equals to 1"

    if oshape is None:
        oshape = []
        for h in hx:
            for hsi in h.shape:
                oshape.append(hsi+g.shape[0]-1)

    out = np.zeros(oshape, dtype=g.dtype)
    g = np.ascontiguousarray(g)
    hx = [np.ascontiguousarray(h) for h in hx]

    libx.outerconv_diagonal_nm(out, g, hx, padleft, stride)
    return out

def outerconv_2d(g, h, stride=1):
    """g @ h
    Only support Order-1 2-dimensional outer convolution

    <-> this equals to convTranspose2D
    """
    assert np.ndim(g) == 2 and np.ndim(h) == 2

    if stride > 1:
        _hs = h.shape
        _hp = _hs[0] * (stride - 1), _hs[1] * (stride - 1)
        h = np.pad(h, [(0, _hp[0]), (0, _hp[1])])\
            .reshape(stride, _hs[0], stride, _hs[1])\
            .transpose(1, 0, 3, 2).reshape(stride*_hs[0], stride*_hs[1])
    
    psg = h.shape[1] - 1
    psh = g.shape[1] - 1

    pg = np.pad(g, [(0, 0), (0, psg)]).reshape(-1)
    ph = np.pad(h, [(0, 0), (0, psh)]).reshape(-1)

    goh = outerconv(pg, ph, stride=1)

    glen = g.shape[0] + h.shape[0]- 1

    R = glen * (g.shape[1] + h.shape[1] - 1)

    return goh[0:R].reshape(glen, -1)

