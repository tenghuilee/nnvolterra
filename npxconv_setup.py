
# Build

from distutils.core import setup, Extension
import numpy as np

# import distutils.sysconfig
# cfg_vars = distutils.sysconfig.get_config_vars()
# cfg_vars["BASECFLAGS"] = ""
# cfg_vars["CFLAGS"] = "-Wall -O0 -g -fwrapv"
# cfg_vars["LDCXXSHARED"] += " -O0 -g "


setup(
    name="libnpxconv",
    version="1.0",
    description="xconvoution for Python Numpy",
    author="Aliy",
    long_description="""order n convolution and outer convolution""",
    headers=["./xconvolution.hpp"],
    ext_modules=[Extension(
        "libnpxconv",
        sources=['npxconv.cpp'],
        define_macros=[
            # ('Py_TRACE_REFS',),
            # ('PY_DEBUG',),
            ('MAJOR_VERSION', '1'),
            ('MINOR_VERSION', '0')],
        include_dirs=['.', np.get_include()],
        # libraries = ['tcl83'],
        # library_dirs = ['/usr/local/lib'],
        # extra_compile_args=["-O0", "-g", "-fopenmp"]
    )]
)
