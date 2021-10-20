r"""
Some Tensor decomposition methods

- CP decomposition
    - TPM: tensor power method
"""

from operator import truediv
import numpy as np
from numpy.core.fromnumeric import ndim
from numpy.core.records import get_remaining_size


def _tpm_core(X: np.ndarray, ridx, factors):
    """
    - X:
    - ridx: rank index
    - factors: list of np.ndarray with shape (rank, s)
    """

    ndimx = np.ndim(X)
    for i in range(ndimx):
        # new shape
        # move dim i to front
        trpx = [i]
        for j in range(ndimx):
            if j == i:
                continue
            trpx.append(j)

        tX = np.transpose(X, axes=trpx)

        # [ndimx-1 ... 0]
        for j in range(ndimx-1, -1, -1):
            if j == i:
                continue
            tX = np.matmul(tX, factors[j][ridx])

        factors[i][ridx] = tX / np.linalg.norm(tX, ord=2)
    
def tensor_power_method(X: np.ndarray, rank=-1, maxiter=400, err=1e-5, truncat=True, verbose=False):
    """
    X <- w1 u1(1) u2(1) ... + w2 u1(2) u2(2)... + ...

    Requires:
    - X: ndarray
    - rank: if <= 0, prod(X.shape) other wise rank
    Return:
    - [w1, w2, ...]
    - [u1, u2, ...]

    References:
    [1] G. I. Allen, “Sparse Higher-Order Principal Components Analysis,” in Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, 2012, vol. 22, pp. 27–36, [Online]. Available: http://proceedings.mlr.press/v22/allen12.html.
    """

    shapeX = X.shape

    if rank <= 0:
        rank = np.prod(shapeX)

    weights = np.zeros(rank)
    factors = []
    for s in shapeX:
        _t = np.random.rand(rank, s)
        _t /= np.linalg.norm(_t, ord=2, axis=(-1), keepdims=True)
        factors.append(_t)

    outX = X.copy()
    for r in range(rank):
        isconverged = True
        for it in range(maxiter):
            fac_hist = [f[r].copy() for f in factors]

            _tpm_core(outX, r, factors)

            isconverged = True
            for fh, f in zip(fac_hist, factors):
                _d = fh - f[r]
                _e = np.mean(_d * _d)
                if _e > err:
                    isconverged = False
                    break

            if isconverged:
                if verbose:
                    print(f"rank {r} converged at iter {it} with err {_e}")
                break
        # out of iter
        if not isconverged:
            if verbose:
                print(f"rank {r} not converged after iter {maxiter} with err {_e}")

        tX = outX
        outerfr = np.ones(1)
        for f in reversed(factors):
            tX = np.matmul(tX, f[r])
            outerfr = np.outer(f[r], outerfr)
            outerfr = np.reshape(outerfr, (-1))

        if tX < 0:
            tX *= -1
            outerfr *= -1
            factors[0][r] *= -1

        weights[r] = tX
        outX -= tX * np.reshape(outerfr, shapeX)
        if truncat and np.linalg.norm(outX) < 1e-8:
            # seems converged
            return weights[0:r+1], [f[0:r+1] for f in factors]

    return weights, factors

def hosvd(X: np.ndarray, truncat=True, sig=False):
    """
    HOSVD: hight order singular vector decomposition
    Tucker decompositin

    Require:
    - X: 
    - tuncat: 
    - outs: if true return the singular value of each dimension

    Return:
    - G
    - [A1, A2, ...]
    """

    Ax = []
    shist = []

    for i in range(X.ndim):

        xaxis = [i]
        for j in range(X.ndim):
            if j == i:
                continue
            xaxis.append(j)
        tX = np.transpose(X, xaxis)
        xshape = list(tX.shape)
        tX = np.reshape(tX, (xshape[0], -1))

        U, S, V = np.linalg.svd(tX, False)

        if truncat:
            m1s = S / np.sum(S)
            _acs = 0.0
            _rank = S.shape[0]
            for k, s in enumerate(m1s):
                _acs += s
                if _acs > 0.999:
                    _rank = k + 1
                    break
            
            U = U[:,0:_rank]
            S = S[0:_rank]
            V = V[0:_rank,:]
        
        xshape[0] = V.shape[0]
        X = np.reshape(S[:,None] * V, xshape)

        Ax.append(U)
        shist.append(S)

    # X.T shape (n1, n2, n3, ...) -> (..., n3, n2, n1)
    if sig:
        return X.T, Ax, shist
    else:
        return X.T, Ax
        

def cp_build(weights: np.ndarray, factors):
    """
    - wieghts: shape (rank)
    - factors: list of [(rank, N), ...]
    """
    xshape = []
    for f in factors:
        xshape.append(f.shape[1])

    ans = np.zeros(xshape)
    for r, w in enumerate(weights):
        _t = np.ones(1)
        for f in reversed(factors):
            _t = np.outer(f[r], _t)
            _t = np.reshape(_t, -1)

        ans += w * np.reshape(_t, xshape)

    return ans


def randn_cp(shape: list, rank: int):
    """
    Randomly generate cp weights and factors

    Require:
    - shape: output shape
    - rank: CP rank, must <= min(shape)

    Return:
    - X: shape
    """
    assert rank <= np.min(shape), "rank must <= min(shape)"

    weights = np.random.rand(rank)
    factors = []
    for s in shape:
        factors.append(np.random.randn(rank, s))

    return cp_build(weights, factors)


def tucker_build(G: np.ndarray, Ax: list):
    """
    compute G x1 A1^T x2 A2^T ...
    """

    for i, A in enumerate(Ax):
        gaxis = [i]
        for j in range(len(Ax)):
            if j == i:
                continue
            gaxis.append(j)
        G = np.transpose(G, gaxis)
        gshape = list(G.shape)
        G = np.matmul(A, G.reshape(gshape[0], -1))
        gshape[0] = A.shape[0]
        G = np.reshape(G, gshape)

        # print(gshape)
    
    return G.T


def randn_tucker(shape: list, rank: list):
    """
    generate random tucker tensors

    Require:
    - shape: output shape
    - rank: list|int
    """
    if isinstance(rank, int):
        rank = [rank for _ in shape]
    else:
        assert len(shape) == len(rank), "length of shape and rank must equal"

    for s,r in zip(shape, rank):
        assert s >= r, f"shape at (..., s={s}, ...) < rank at (..., r={r}, ...)"
    
    G = np.random.randn(*rank)

    Ax = [np.random.randn(s, r) for s, r in zip(shape, rank)]

    return tucker_build(G, Ax)

