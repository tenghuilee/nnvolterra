#include <Python.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/ufuncobject.h>

#include "xconvolution.hpp"

#define MY_PY_ASSERT(expr, __exp_point)                                               \
  {                                                                                   \
    if (!static_cast<bool>(expr)) {                                                   \
      PyErr_SetObject(                                                                \
          PyExc_AssertionError,                                                       \
          PyBytes_FromFormat(                                                         \
              "assert faild: at file %s:%d reasion: %s", __FILE__, __LINE__, #expr)); \
      goto __exp_point;                                                               \
    }                                                                                 \
  }

// reference https://numpy.org/doc/stable/user/c-info.how-to-extend.html

// PyObject* ping(PyObject* self) {
//   printf("pong\n");
//   Py_RETURN_NONE;
// }

// PyObject* test(PyObject* self, PyObject* args) {  //

//   PyArrayObject *ker, *src, *out;

//   if (!PyArg_ParseTuple(args, "OOO", &ker, &src, &out)) {
//     PyErr_SetString(PyExc_ValueError, "Input variable is invalid");
//     return NULL;
//   }

//   IS_ND_ARRAY(ker, 1);
//   IS_ND_ARRAY(src, 1);

//   int len = PyArray_DIM(out, 0);
//   if (PyArray_TYPE(ker) == NPY_FLOAT32) {
//     auto kerp = (npy_float32*)PyArray_DATA(ker);
//     auto srcp = (npy_float32*)PyArray_DATA(src);
//     auto outp = (npy_float32*)PyArray_DATA(out);

//     for (int i = 0; i < len; i++) {  //
//       outp[i] = kerp[i] + srcp[i];
//     }
//   } else if (PyArray_TYPE(ker) == NPY_FLOAT64) {
//     auto kerp = (npy_float64*)PyArray_DATA(ker);
//     auto srcp = (npy_float64*)PyArray_DATA(src);
//     auto outp = (npy_float64*)PyArray_DATA(out);

//     for (int i = 0; i < len; i++) {  //
//       outp[i] = kerp[i] + srcp[i];
//     }
//   }

//   Py_RETURN_NONE;
// }

/**
 * order = -1 => invalid
 */
struct ConvArgs {
  PyArrayObject* dest;    // destination
  PyArrayObject* kernel;  // kernel
  PyObject* srcx;         // a np.ndarray or a list of np.ndarray
  PyObject* padleft;      // pad left
  int np_dtype;           // np.ndarray dtype; micro start with NPY_XXX
  int stride;             // stride; default 1
  int order;              // order -1 => invalid; sould be 1, ...
  void** c_src;           // a list of pointer to src
  long* c_srclen;         // size of each src; | size src0 | size src1 | ...
  int* c_srcndim;         // ndim of each src; sum of this is the lenght of `c_src_len`
  void* c_dest;           // pointer to dest
  long* c_dstlen;         // pointer to size of dest; NOT malloc Variable!!!
  void* c_kernel;         // pointer to kernel
  long* c_kerlen;         // pointer to size of kernel; NOT malloc Variable!!!
  int* c_pad;             // pad
};

ConvArgs new_ConvArgs(PyObject* args, int isdiagonal = false);
void del_ConvArgs(ConvArgs* self);

ConvArgs new_ConvArgs(PyObject* args, int isdiagonal) {
  ConvArgs self = {
      .dest = NULL,
      .kernel = NULL,
      .srcx = NULL,
      .padleft = PyInt_FromLong(0),
      .stride = 1,
      .order = -1,
      .c_src = NULL,
      .c_srclen = NULL,
      .c_srcndim = NULL,
      .c_dest = NULL,
      .c_dstlen = NULL,
      .c_kernel = NULL,
      .c_kerlen = NULL,
      .c_pad = NULL,
  };
  int all_ndims = 1;

  if (!PyArg_ParseTuple(
          args,
          "OOO|Oi",
          &self.dest,
          &self.kernel,
          &self.srcx,
          &self.padleft,
          &self.stride)) {
    PyErr_SetString(
        PyExc_ValueError,
        "Invalid input. require (dest, kernel, srcx, /, padleft, stride)");
    return self;
  }
  // special case at outer_convolution_diagonal
  if (isdiagonal) {
    MY_PY_ASSERT(PyList_CheckExact(self.srcx), __error_return);
    self.order = PyList_Size(self.srcx);
  } else {
    self.order = PyArray_NDIM(self.kernel);
  }
  self.np_dtype = PyArray_TYPE(self.dest);
  MY_PY_ASSERT(PyArray_TYPE(self.kernel) == self.np_dtype, __error_return);

  self.c_src = (void**)malloc(sizeof(void*) * self.order);
  self.c_srcndim = (int*)malloc(sizeof(int) * self.order);
  // self.c_pad will be init soon
  // self.c_srclen will be init soon

  // this are NOT malloc Variable!!!
  self.c_dest = PyArray_DATA(self.dest);
  self.c_dstlen = PyArray_DIMS(self.dest);
  self.c_kernel = PyArray_DATA(self.kernel);
  self.c_kerlen = PyArray_DIMS(self.kernel);

  // is np.ndarray => all src is the same
  if (PyArray_CheckExact(self.srcx)) {
    PyArrayObject* _px = (PyArrayObject*)self.srcx;
    MY_PY_ASSERT(PyArray_TYPE(_px) == self.np_dtype, __error_return);
    int _ndim = PyArray_NDIM(_px);
    all_ndims = self.order * _ndim;
    self.c_srclen = (long*)malloc(sizeof(long) * all_ndims);

    long* _psrclen = self.c_srclen;

    for (int i = 0; i < self.order; i++) {
      self.c_src[i] = PyArray_DATA(_px);
      self.c_srcndim[i] = _ndim;

      // self.c_srclen is
      // | s0, s1, ... | s0, s1, ... | ...
      for (int j = 0; j < _ndim; j++) {
        *_psrclen = PyArray_DIM(_px, j);
        _psrclen++;
      }
    }
  } else if (PyList_CheckExact(self.srcx)) {
    // is list of np.ndarray; len must equals to order
    MY_PY_ASSERT(PyList_Size(self.srcx) == self.order, __error_return);

    all_ndims = 1;
    for (int i = 0; i < self.order; i++) {
      PyObject* _px = PyList_GetItem(self.srcx, i);
      MY_PY_ASSERT(
          PyArray_Check(_px) && PyArray_TYPE((PyArrayObject*)_px) == self.np_dtype,
          __error_return);

      int _ndim = PyArray_NDIM((PyArrayObject*)_px);
      self.c_srcndim[i] = _ndim;
      all_ndims += _ndim;
    }

    self.c_srclen = (long*)malloc(sizeof(long) * all_ndims);

    long* _psrclen = self.c_srclen;

    // copy to each i
    for (int i = 0; i < self.order; i++) {
      PyArrayObject* _px = (PyArrayObject*)PyList_GetItem(self.srcx, i);
      self.c_src[i] = PyArray_DATA(_px);
      for (int j = 0; j < self.c_srcndim[i]; j++) {
        *_psrclen = PyArray_DIM(_px, j);
        _psrclen++;
      }
    }
  } else {
    PyErr_SetString(
        PyExc_ValueError,
        "Invalid src; require np.ndarray or list of np.ndarray (len = order)");
    goto __error_return;
  }

  self.c_pad = (int*)malloc(sizeof(int) * all_ndims);

  // handle padleft
  if (PyInt_Check(self.padleft)) {
    // if int copy to all
    int _t = (int)PyInt_AsLong(self.padleft);
    for (int i = 0; i < all_ndims; i++) { self.c_pad[i] = _t; }
  } else {
    if (PyArray_Check(self.padleft)) {
      // convert to python list
      self.padleft = PyArray_ToList((PyArrayObject*)self.padleft);
    }

    if (!PyList_Check(self.padleft)) {
      //
      PyErr_SetString(
          PyExc_ValueError, "Invalid pad; require int or list of int (len=order)");
      goto __error_return;
    }

    int pidx = 0;
    for (int i = 0; i < PyList_Size(self.padleft) && pidx < all_ndims; i++) {
      self.c_pad[pidx] = (int)PyInt_AsLong(PyList_GetItem(self.padleft, i));
      pidx++;
    }
    // write 0 to all the end
    while (pidx < all_ndims) {
      self.c_pad[pidx] = 0;
      pidx++;
    }
  }

  return self;
__error_return:
  del_ConvArgs(&self);
  return self;
}

void del_ConvArgs(ConvArgs* self) {
  if (self->c_src) {
    free(self->c_src);
    self->c_src = NULL;
  }

  if (self->c_pad) {
    free(self->c_pad);
    self->c_pad = NULL;
  }

  if (self->c_srcndim) {
    free(self->c_srcndim);
    self->c_srcndim = NULL;
  }

  if (self->c_srclen) {
    free(self->c_srclen);
    self->c_srclen = NULL;
  }

  memset(self, 0, sizeof(ConvArgs));
  self->order = -1;
}

PyObject* conv1d_order_n(PyObject* self, PyObject* args) {
  ConvArgs carg = new_ConvArgs(args);
  if (carg.order < 0) { Py_RETURN_NONE; }

  if (carg.np_dtype == NPY_FLOAT32) {
    convolution1d_order_n(
        carg.order,
        (npy_float32*)carg.c_dest,
        carg.c_dstlen[0],
        (const npy_float32*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float32**)carg.c_src,
        carg.c_srclen,
        carg.c_pad,
        carg.stride);
  } else if (carg.np_dtype == NPY_FLOAT64) {
    convolution1d_order_n(
        carg.order,
        (npy_float64*)carg.c_dest,
        carg.c_dstlen[0],
        (const npy_float64*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float64**)carg.c_src,
        carg.c_srclen,
        carg.c_pad,
        carg.stride);
  }

  del_ConvArgs(&carg);
  Py_RETURN_NONE;
}

PyObject* conv1d_extend(PyObject* self, PyObject* args) {
  ConvArgs carg = new_ConvArgs(args);
  if (carg.order < 0) { Py_RETURN_NONE; }

  if (carg.np_dtype == NPY_FLOAT32) {
    convolution1d_extend(
        carg.order,
        (npy_float32*)carg.c_dest,
        carg.c_dstlen[0],
        (const npy_float32*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float32**)carg.c_src,
        carg.c_srclen,
        carg.c_pad,
        carg.stride);
  } else if (carg.np_dtype == NPY_FLOAT64) {
    convolution1d_extend(
        carg.order,
        (npy_float64*)carg.c_dest,
        carg.c_dstlen[0],
        (const npy_float64*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float64**)carg.c_src,
        carg.c_srclen,
        carg.c_pad,
        carg.stride);
  }

  del_ConvArgs(&carg);
  Py_RETURN_NONE;
}

PyObject* outerconv_211(PyObject* self, PyObject* args) {
  ConvArgs carg = new_ConvArgs(args);
  if (carg.order < 0) { Py_RETURN_NONE; }

  if (carg.np_dtype == NPY_FLOAT32) {
    outer_convolution_211(
        (npy_float32*)carg.c_dest,
        carg.c_dstlen,
        (const npy_float32*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float32**)carg.c_src,
        carg.c_srclen,
        carg.c_pad,
        carg.stride);
  } else if (carg.np_dtype == NPY_FLOAT64) {
    outer_convolution_211(
        (npy_float64*)carg.c_dest,
        carg.c_dstlen,
        (const npy_float64*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float64**)carg.c_src,
        carg.c_srclen,
        carg.c_pad,
        carg.stride);
  }

  del_ConvArgs(&carg);
  Py_RETURN_NONE;
}

PyObject* outerconv_nm(PyObject* self, PyObject* args) {
  ConvArgs carg = new_ConvArgs(args);
  if (carg.order < 0) { Py_RETURN_NONE; }

  if (carg.np_dtype == NPY_FLOAT32) {
    outer_convolution_nm(
        carg.order,
        (npy_float32*)carg.c_dest,
        carg.c_dstlen,
        (const npy_float32*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float32**)carg.c_src,
        carg.c_srclen,
        carg.c_srcndim,
        carg.c_pad,
        carg.stride);
  } else if (carg.np_dtype == NPY_FLOAT64) {
    outer_convolution_nm(
        carg.order,
        (npy_float64*)carg.c_dest,
        carg.c_dstlen,
        (const npy_float64*)carg.c_kernel,
        carg.c_kerlen,
        (const npy_float64**)carg.c_src,
        carg.c_srclen,
        carg.c_srcndim,
        carg.c_pad,
        carg.stride);
  }

  del_ConvArgs(&carg);
  Py_RETURN_NONE;
}

PyObject* outerconv_diagonal_nm(PyObject* self, PyObject* args) {
  ConvArgs carg = new_ConvArgs(args, true);
  if (carg.order < 0) { Py_RETURN_NONE; }

  if (carg.np_dtype == NPY_FLOAT32) {
    outer_convolution_diagonal_nm(
        carg.order,
        (npy_float32*)carg.c_dest,
        carg.c_dstlen,
        (const npy_float32*)carg.c_kernel,
        carg.c_kerlen[0],
        (const npy_float32**)carg.c_src,
        carg.c_srclen,
        carg.c_srcndim,
        carg.c_pad,
        carg.stride);
  } else if (carg.np_dtype == NPY_FLOAT64) {
    outer_convolution_diagonal_nm(
        carg.order,
        (npy_float64*)carg.c_dest,
        carg.c_dstlen,
        (const npy_float64*)carg.c_kernel,
        carg.c_kerlen[0],
        (const npy_float64**)carg.c_src,
        carg.c_srclen,
        carg.c_srcndim,
        carg.c_pad,
        carg.stride);
  }

  del_ConvArgs(&carg);
  Py_RETURN_NONE;
}

#define CONV_REQUIRE_COMMEN                                                     \
  "Require:\n"                                                                  \
  " - out (np.ndarray): output signal\n"                                        \
  " - ker (np.ndarray): convolution kernel\n"                                   \
  " - src (list|np.ndarray): source signal\n"                                   \
  "   1. list: the length <=> the order of convolution\n"                       \
  "   2. np.ndarray: the length of convolution is the size "                    \
  " - padleft (int|list): pad left (default 0), if is a list, its length must " \
  "equals `n`\n"                                                                \
  " - stride (int): (default 1)\n"

PyMODINIT_FUNC PyInit_libnpxconv(void) {
  // important!! activate numpy
  import_array();
  if (PyErr_Occurred()) {
    printf("Failed to load numpy\n");
    return NULL;
  }

  static PyMethodDef func[] = {
      // {"ping", (PyCFunction)ping, METH_NOARGS, "ping()\n"},
      {"conv1d_order_n",
       (PyCFunction)conv1d_order_n,
       METH_VARARGS,
       "order n convolution\n"
       " if 0 < n <= 3, naive implementation \n"
       " if 3 < n, recursion implementation \n" CONV_REQUIRE_COMMEN},
      {"conv1d_extend",
       (PyCFunction)conv1d_extend,
       METH_VARARGS,
       "order n convolution, using recursion\n" CONV_REQUIRE_COMMEN},
      {"outerconv_211",
       (PyCFunction)outerconv_211,
       METH_VARARGS,
       "order n convolution, using recursion\n" CONV_REQUIRE_COMMEN},
      {"outerconv_nm",
       (PyCFunction)outerconv_nm,
       METH_VARARGS,
       "outer conv nmw, recursion\n" CONV_REQUIRE_COMMEN},
      {"outerconv_diagonal_nm",
       (PyCFunction)outerconv_diagonal_nm,
       METH_VARARGS,
       "outer convolution for diagonal kernel, where kernel is 1D "
       "signal." CONV_REQUIRE_COMMEN},
      {NULL, NULL, 0, NULL},  // sentinel
  };

  static struct PyModuleDef xconv = {
      PyModuleDef_HEAD_INIT,
      "xconvolution for numpy",
      "useage:\n",
      -1,
      func,
  };
  return PyModule_Create(&xconv);
}
