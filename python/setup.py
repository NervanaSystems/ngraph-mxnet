# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""
from __future__ import absolute_import
import os
import platform
import sys
import glob

from setuptools import find_packages
# need to use distutils.core for correct placement of cython dll
kwargs = {}
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension
    kwargs = {'install_requires': ['numpy<=1.15.2,>=1.8.2', 'requests<2.19.0,>=2.18.4', 'graphviz<0.9.0,>=0.8.1'], 'zip_safe': False}

with_cython = False
if '--with-cython' in sys.argv:
    with_cython = True
    sys.argv.remove('--with-cython')

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATHS = libinfo['find_lib_path']()
LIB_PATHS += glob.glob("/".join(LIB_PATHS[0].split("/")[0:-1]) + "/*.so*")

# mxnet assumes the .so files are located in the same directory
# as the python module. To get wheel to package things that way,
# we link all of the .so files into the python directory.
symlinks = []
for src in set(LIB_PATHS):
  symlinks.append('mxnet/' + src.split('/')[-1])
  os.symlink(src, symlinks[-1])

__version__ = libinfo['__version__']

sys.path.insert(0, CURRENT_DIR)

# Try to generate auto-complete code
try:
    from mxnet.base import _generate_op_module_signature
    from mxnet.ndarray.register import _generate_ndarray_function_code
    from mxnet.symbol.register import _generate_symbol_function_code
    _generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
    _generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)
except: # pylint: disable=bare-except
    pass

def config_cython():
    """Try to configure cython and return cython configuration"""
    if not with_cython:
        return []
    # pylint: disable=unreachable
    if os.name == 'nt':
        print("WARNING: Cython is not supported on Windows, will compile without cython module")
        return []

    try:
        from Cython.Build import cythonize
        # from setuptools.extension import Extension
        if sys.version_info >= (3, 0):
            subdir = "_cy3"
        else:
            subdir = "_cy2"
        ret = []
        path = "mxnet/cython"
        if os.name == 'nt':
            library_dirs = ['mxnet', '../build/Release', '../build']
            libraries = ['libmxnet']
        else:
            library_dirs = None
            libraries = None

        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "mxnet/%s/.%s" % (subdir, fn[:-4]),
                ["mxnet/cython/%s" % fn],
                include_dirs=["../include/", "../3rdparty/tvm/nnvm/include"],
                library_dirs=library_dirs,
                libraries=libraries,
                language="c++"))
        return cythonize(ret)
    except ImportError:
        print("WARNING: Cython is not installed, will compile without cython module")
        return []

# Create a custom wheel class to add information on what kind of 
# platforms/python versions are supported. 
# Unfortunately, it's fairly generic on what linux/cpu versions we support, but 
# This matches the ngraph_tf wheel naming scheme
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False
    def get_tag(self):
        _, _, plat = _bdist_wheel.get_tag(self)
        # let users know this is a py2 and py3 compatible package, but only linux x86_64
        return ('py2.py3', 'none', plat)

setup(name='ngraph-mxnet',
      version="0.5.0rc0",
      description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      packages=find_packages(),
      package_data={"mxnet":  "*.so*"}, # tell the wheel to include all of the .so files in the mxnet module
      url='https://github.com/NervanaSystems/ngraph-mxnet',
      ext_modules=config_cython(),
      cmdclass={'bdist_wheel': bdist_wheel},
      classifiers=[
          # https://pypi.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Other',  # R, Scala
          'Programming Language :: Perl',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      **kwargs)

# remove the temporary simlinks to clean up the directory
for link in symlinks:
  os.remove(link)
