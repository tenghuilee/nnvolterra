#ifndef __H_X_CONVOLUTION_H
#define __H_X_CONVOLUTION_H

#include <assert.h>
#include <math.h>
#include <memory.h>

template <typename T>
void convolution1d_order1(
    T* dest,
    const long dstlen,
    const T* kernel,
    const long kerlen,
    const T* src,
    const long srclen,
    const int padleft,
    const int stride) {
  T* desti = dest;
  long ndil = kerlen;

  // this is a dangous opeartion
  assert((long)src > (long)padleft);
  const T* __src = src - (long)padleft;  // left pad
  const T* src__ = src + (long)srclen;   // right pad

  for (long t = 0, ts = 0; t < dstlen; t++, ts += stride) {
    const T* ki = kernel;

    const T* xt = &__src[ts];
    // to loops
    T yt = 0.0;
    for (long i = 0; i < ndil; i++, ki++) {  //
      const T* xti = &xt[i];

      if (xti < src || xti >= src__ || xti[0] == 0) {  //
        continue;
      }
      yt += (*ki) * (*xti);
    }
    *desti++ = yt;
  }
}

/**
 *
 * x(t) <- sum_{i,j} h_{ij} x(t+i) y(t+j)
 *
 * - both i, j is range from 0 to kerlen-1
 */
template <typename T>
void convolution1d_order2_n(
    T* dest,
    const long dstlen,
    const T* kernel,
    const long* kerlen,
    const T** srcx,
    const long* srclen,
    const int* padleft,
    const int stride) {
  T* desti = dest;
  // TODO: pad and length
  long ndil[2] = {
      kerlen[0],
      kerlen[1],
  };

  const T* __srcx[2] = {
      srcx[0] - padleft[0],
      srcx[1] - padleft[1],
  };

  const T* srcx__[2] = {
      srcx[0] + srclen[0],
      srcx[1] + srclen[1],
  };

  for (long t = 0, ts = 0; t < dstlen; t++, ts += stride) {
    const T* ki = kernel;
    const T* xt1 = __srcx[0] + ts;
    const T* xt2 = __srcx[1] + ts;

    // to loops
    T yt = 0.0;
    for (int i = 0; i < ndil[0]; i++) {
      const T* xt1i = xt1 + i;
      if (xt1i < srcx[0] || xt1i >= srcx__[0] || xt1i[0] == 0) {  //
        ki += kerlen[1];
        continue;
      }
      for (int j = 0; j < ndil[1]; j++) {  //
        const T* xt2j = xt2 + j;
        if (xt2j < srcx[1] || xt2j >= srcx__[1] || xt2j[0] == 0) {
          ki++;
          continue;
        }
        yt += (*ki++) * xt1i[0] * xt2j[0];
      }
    }
    *desti = yt;
    desti++;
  }
}

template <typename T>
void convolution1d_order3_n(
    T* dest,
    const long dstlen,
    const T* kernel,
    const long* kerlen,
    const T** srcx,
    const long* srclen,
    const int* padleft,
    const int stride) {
  T* desti = dest;

  const int ndil[3] = {
      (int)kerlen[0],
      (int)kerlen[1],
      (int)kerlen[2],
  };

  const T* __srcx[3] = {
      srcx[0] - padleft[0],
      srcx[1] - padleft[1],
      srcx[2] - padleft[2],
  };

  const T* srcx__[3] = {
      srcx[0] + srclen[0],
      srcx[1] + srclen[1],
      srcx[2] + srclen[2],
  };

  for (long t = 0, ts = 0; t < dstlen; t++, ts += stride) {
    const T* xt[3] = {
        __srcx[0] + ts,
        __srcx[1] + ts,
        __srcx[2] + ts,
    };

    // to loops
    T yt = 0.0;
    auto ki = kernel;
    for (int i = 0; i < ndil[0]; i++) {
      const T* _xtip = xt[0] + i;
      if (_xtip < srcx[0] || _xtip >= srcx__[0] || _xtip[0] == 0) {
        ki += kerlen[1] * kerlen[2];
        continue;
      }

      T _xti = *_xtip;

      // dim 2
      for (int j = 0; j < ndil[1]; j++) {
        _xtip = xt[1] + j;
        if (_xtip < srcx[1] || _xtip >= srcx__[1] || _xtip[0] == 0) {
          ki += kerlen[2];
          continue;
        }
        T _xtij = _xti * (*_xtip);

        for (int k = 0; k < ndil[2]; k++, ki++) {  //
          _xtip = xt[2] + k;

          if (_xtip < srcx[2] || _xtip >= srcx__[2] || _xtip[0] == 0) { continue; }
          yt += (*ki) * _xtij * (*_xtip);
          // printf("bse 3 %f %f %f\n", yt, kerij[k], _xti * (*_xtip));
        }
      }
    }
    *desti = yt;
    desti++;
  }
}

template <typename T>
T __convolution1d_extend_core(
    const int n,
    const int depth,
    const T** srcx,
    const T** srcx__,
    const T xti,
    const T** xt,
    const T* kernel,
    const int* ks_table,
    const int* ndil) {
  const T* ker = kernel;

  T yt = 0.0;
  for (int i = 0; i < ndil[depth]; i++, ker += ks_table[depth]) {
    const T* _xtip = xt[depth] + i;
    // out of bound
    if (_xtip < srcx[depth] || _xtip >= srcx__[depth] || _xtip[0] == 0) {
      // printf("[%d/%d], skipped because %d %d\n", depth, n, _xtip < srcx[depth], _xtip
      // >= srcx__[depth]);
      continue;
    }

    T local_xti = (depth == 0) ? (*_xtip) : xti * (*_xtip);

    // reach last dim
    if (depth == n - 1) {
      yt += (*ker) * local_xti;
      // printf("ext %d %f %f %f\n", n, yt, *ker, local_xti);
    } else {
      yt += __convolution1d_extend_core(
          n, depth + 1, srcx, srcx__, local_xti, xt, ker, ks_table, ndil);
    }
  }
  return yt;
}

template <typename T>
void convolution1d_extend(
    const int n,
    T* dest,
    const long dstlen,
    const T* kernel,
    const long* kerlen,
    const T** srcx,
    const long* srclen,
    const int* padleft,
    const int stride) {
  T* desti = dest;
  int* ndil = (int*)alloca(sizeof(int) * n);
  const T** __srcx = (const T**)alloca(sizeof(T*) * n);
  const T** srcx__ = (const T**)alloca(sizeof(T*) * n);
  const T** xt = (const T**)alloca(sizeof(T*) * n);
  int* ks_table = (int*)alloca(sizeof(int) * n);

  for (int i = n - 1; i >= 0; i--) {
    ndil[i] = kerlen[i];
    __srcx[i] = srcx[i] - padleft[i];
    srcx__[i] = srcx[i] + srclen[i];

    assert(__srcx[i] > 0);
    assert(srcx__[i] > 0);

    // ks_table[n-1] = 1
    // ks_table[n-2] = ks[n-1]
    // ks_table[n-3] = ks[n-2] ks_table[n-2]
    // ks_table[n-4] = ks[n-3] ks_table[n-3]
    // ......
    if (i == n - 1) {
      ks_table[i] = 1;
    } else {
      ks_table[i] = kerlen[i + 1] * ks_table[i + 1];
    }
  }

  for (long t = 0, ts = 0; t < dstlen; t++, ts += stride) {
    for (int i = 0; i < n; i++) {  //
      xt[i] = __srcx[i] + ts;
    }

    // to loops
    *desti =
        __convolution1d_extend_core(n, 0, srcx, srcx__, (T)1, xt, kernel, ks_table, ndil);
    desti++;
  }
}

template <typename T>
void convolution1d_order_n(
    int n,
    T* dest,
    const long dstlen,
    const T* kernel,
    const long* kerlen,
    const T** srcx,
    const long* srclen,
    const int* padleft,
    const int stride) {
  switch (n) {
    case 1:
      convolution1d_order1(
          dest, dstlen, kernel, kerlen[0], srcx[0], srclen[0], padleft[0], stride);
      break;
    case 2:
      convolution1d_order2_n(dest, dstlen, kernel, kerlen, srcx, srclen, padleft, stride);
      break;
    case 3:
      convolution1d_order3_n(dest, dstlen, kernel, kerlen, srcx, srclen, padleft, stride);
      break;
    default:
      convolution1d_extend(
          n, dest, dstlen, kernel, kerlen, srcx, srclen, padleft, stride);
      break;
  }
}

/**
 *
 * outer convolution for 2d kernel and [1d signal, 1d signal]
 *
 * @param dest: destination
 * @param dstlen: size of dest
 * @param kernel: the 2d kernel
 * @param kernel: size of kernel
 * @param srcx: source signal
 * @param srclen:
 * @param padleft:
 * @param stride:
 */
template <typename T>
void outer_convolution_211(
    T* dest,
    const long* dstlen,
    const T* kernel,
    const long* kerlen,
    const T** srcx,
    const long* srclen,
    const int* padleft,
    const int stride) {
  // begin; shift left
  const T* __src[2] = {
      srcx[0] - padleft[0],
      srcx[1] - padleft[1],
  };
  assert(__src[0] > 0 && __src[1] > 0);
  // end
  const T* src__[2] = {
      srcx[0] + srclen[0],
      srcx[1] + srclen[1],
  };

  T* dst = dest;
  for (long t0 = 0, ts0 = 0; t0 < dstlen[0]; t0++, ts0 += stride, dst += dstlen[1]) {
    const T* _xts0 = __src[0] + ts0;
    for (long t1 = 0, ts1 = 0; t1 < dstlen[1]; t1++, ts1 += stride) {
      const T* _xts1 = __src[1] + ts1;
      // convolution
      T ans = 0.0;
      const T* ker = kernel;
      for (long i0 = 0; i0 < kerlen[0]; i0++, ker += kerlen[1]) {
        const T* _xtsi0 = _xts0 - i0;
        if (_xtsi0 < srcx[0] || _xtsi0 >= src__[0] || _xtsi0[0] == 0) {
          // out of bound or is zero
          continue;
        }

        for (long i1 = 0; i1 < kerlen[1]; i1++) {
          const T* _xtsi1 = _xts1 - i1;
          if (_xtsi1 < srcx[1] || _xtsi1 >= src__[1] || _xtsi0[1] == 0) {
            // out of bound or is zero
            continue;
          }
          ans += ker[i1] * _xtsi0[0] * _xtsi1[0];
          // printf("[211] ker %f, xt %f\n", ker[i1], _xtsi0[0] * _xtsi1[0]);
        }
      }
      // write to dest
      dst[t1] = ans;
    }
    // step to next
  }
}

template <typename T>
T __outer_convolution_nm_core(
    const int n,
    const int depth,
    const T** srcx,        // src
    const int* pad_left,   //
    const int* dst_idx,    // length ndim
    const long* src_len,   // end point
    const int* src_ndim,   // src ndim
    const long* xs_table,  // x size table
    const T xt,
    const T* kernel,
    const long* kerlen,
    const long* ks_table,
    const int stride) {
  T ans = 0.0;

  const T* ker = kernel;
  const int ndim = src_ndim[depth];
  int flag = 0;

  for (long i = 0, ist = 0; i < kerlen[depth]; i++, ist+=stride, ker += ks_table[depth]) {
    const T* _xtsi = srcx[depth];
    flag = 0;
    // compute index x
    for (int j = 0; j < ndim; j++) {
      long _idx = dst_idx[j] - ist - pad_left[j];
      if (_idx < 0 || _idx >= src_len[j]) {
        // out of bound
        flag = 1;
        break;
      }
      _xtsi += _idx * xs_table[j];
    }
    if (flag || _xtsi[0] == 0) { continue; }

    T local_xt = (depth == 0) ? _xtsi[0] : xt * _xtsi[0];

    if (depth == n - 1) {
      ans += (*ker) * local_xt;
    } else {
      ans += __outer_convolution_nm_core(
          n,
          depth + 1,
          srcx,
          pad_left + ndim,
          dst_idx + ndim,
          src_len + ndim,
          src_ndim,
          xs_table + ndim,
          local_xt,
          ker,
          kerlen,
          ks_table,
          stride);
    }
  }

  return ans;
}
/**
 *
 * outer convolution for nd kernel and [nd signal, nd signal]
 *
 * @param dest: destination
 * @param dstlen: size of dest
 * @param kernel: the 2d kernel
 * @param kernel: size of kernel
 * @param srcx: source signal
 * @param srclen:
 * @param padleft:
 * @param stride:
 */
template <typename T>
void outer_convolution_nm(
    int n,
    T* dest,
    const long* dstlen,
    const T* kernel,
    const long* kerlen,
    const T** srcx,
    const long* srclen,
    const int* srcndim,
    const int* padleft,
    const int stride) {
  // begin; shift left
  int all_ndim = 0;
  for (int i = 0; i < n; i++) { all_ndim += srcndim[i]; }
  long dst_size = 1;
  for (int i = 0; i < all_ndim; i++) { dst_size *= dstlen[i]; }

  int* dst_idx = (int*)alloca(sizeof(int) * all_ndim);      // src index
  int* dst_len = (int*)alloca(sizeof(int) * all_ndim);      //
  long* xs_table = (long*)alloca(sizeof(long) * all_ndim);  // x size table
  long* ks_table = (long*)alloca(sizeof(long) * n);         // kernel size table
  memset(dst_idx, 0, sizeof(int) * all_ndim);

  const long* xs_len = srclen;
  long* pxs_table = xs_table;

  // build xs table
  for (int i = 0; i < n; i++) {
    int ndimi = srcndim[i];
    pxs_table[ndimi - 1] = 1;
    for (int j = ndimi - 2; j >= 0; j--) {  //
      pxs_table[j] = xs_len[j + 1] * pxs_table[j + 1];
    }
    xs_len += ndimi;
    pxs_table += ndimi;
  }

  // build ks table
  // ks_table[n-1] = 1
  // ks_table[n-2] = ks[n-1]
  // ks_table[n-3] = ks[n-2] ks_table[n-2]
  // ks_table[n-4] = ks[n-3] ks_table[n-3]
  // ......
  ks_table[n - 1] = 1;
  for (int i = n - 2; i >= 0; i--) {  //
    ks_table[i] = kerlen[i + 1] * ks_table[i + 1];
  }

  for (int i = 0; i < all_ndim; i++) {  //
    dst_len[i] = dstlen[i];
  }

  for (long t = 0; t < dst_size; t++) {
    // compute
    dest[t] = __outer_convolution_nm_core(
        n,
        0,
        srcx,
        padleft,
        dst_idx,
        srclen,
        srcndim,
        xs_table,
        (T)1,
        kernel,
        kerlen,
        ks_table,
        stride);

    // compute next index
    dst_idx[all_ndim - 1] += 1;
    for (int i = all_ndim - 1; i > 0; i--) {
      if (dst_idx[i] >= dst_len[i]) {
        dst_idx[i] -= dst_len[i];
        dst_idx[i - 1] += 1;
      } else {
        break;
      }
    }
  }
}

/**
 * outer convolution diagonal n1
 *
 * the kernel is 1D signal !!!
 *
 * we will first transfom this 1D signal to a super diagonal tensor
 * y[i,i,i...] = x[i]; other wise 0
 * and then apply outer_convolution_nm
 *
 * and good news is the ouput is super symmetry (<- since the super diagonal tensor is
 * symmetry)
 *
 * @param
 *
 *
 */
template <typename T>
void outer_convolution_diagonal_nm(
    const int n,
    T* dest,
    const long* dstlen,
    const T* kernel,
    const int kerlen,
    const T** srcx,
    const long* srclen,
    const int* srcndim,
    const int* padleft,
    const int stride) {
  // begin; shift left
  int all_ndim = 0;
  for (int i = 0; i < n; i++) { all_ndim += srcndim[i]; }
  long dst_size = 1;
  for (int i = 0; i < all_ndim; i++) { dst_size *= dstlen[i]; }

  int* dst_idx = (int*)alloca(sizeof(int) * all_ndim);      // dest index
  int* dst_len = (int*)alloca(sizeof(int) * all_ndim);      // dst len
  long* xs_table = (long*)alloca(sizeof(long) * all_ndim);  // x size table
  int* ker_stride = (int*)alloca(sizeof(int) * kerlen);

  memset(dst_idx, 0, sizeof(int) * all_ndim);

  const long* xs_len = srclen;
  long* pxs_table = xs_table;
  int* pdst_idx = dst_idx;
  const int* xp_left = padleft;

  for (int i = 0; i < n; i++) {
    int ndimi = srcndim[i];
    pxs_table[ndimi - 1] = 1;
    for (int j = ndimi - 2; j >= 0; j--) {  //
      pxs_table[j] = xs_len[j + 1] * pxs_table[j + 1];
    }
    xs_len += ndimi;
    pxs_table += ndimi;
  }

  // init len
  for (int i = 0; i < all_ndim; i++) {  //
    dst_len[i] = dstlen[i];
  }

  // init ker_stride
  for (int i = 0; i < kerlen; i++) {  //
    ker_stride[i] = i * stride;
  }

  for (long t = 0; t < dst_size; t++) {
    // compute
    T yt = 0.0;
    for (int k = kerlen - 1; k >= 0; k--) {
      // compute src[i] at index ( p... - k )
      pdst_idx = dst_idx;
      xs_len = srclen;
      xp_left = padleft;
      pxs_table = xs_table;

      int flag = 0;
      T x_prod = 1;
      for (int i = 0; i < n; i++,
               pdst_idx += srcndim[i],
               xs_len += srcndim[i],
               xp_left += srcndim[i],
               pxs_table += srcndim[i]) {
        const T* xi = srcx[i];
        for (int j = 0; j < srcndim[i]; j++) {
          // compute index
          long _idx = pdst_idx[j] - ker_stride[k] - xp_left[j];
          if (_idx < 0 || _idx >= xs_len[j]) {
            // out of bound
            flag = 1;
            break;
          }
          xi += _idx * pxs_table[j];
        }
        if (flag || xi[0] == 0) {
          flag = 1;
          break;
        }
        x_prod *= xi[0];
      }
      if (!flag) {  //
        yt += x_prod * kernel[k];
      }
    }

    dest[t] = yt;

    // compute next index
    dst_idx[all_ndim - 1] += 1;
    for (int i = all_ndim - 1; i > 0; i--) {
      if (dst_idx[i] >= dst_len[i]) {
        dst_idx[i] -= dst_len[i];
        dst_idx[i - 1] += 1;
      } else {
        break;
      }
    }
  }
}

#endif  //__H_X_CONVOLUTION_H
