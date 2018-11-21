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

# coding: utf-8
"""Information about mxnet."""
from __future__ import absolute_import
import os
import platform
import logging

def _ancestor_dir(pathname, num_levels):
    for i in range(num_levels):
        pathname = os.path.dirname(pathname)
    return pathname

def get_postbuild_so_dir():
    """
    Assume that MXnet was built in-source, and that this file (libinfo.py) is at its usual location
    within that source tree.
    Return the absolute path of the directory that should contain all of the shared objects
    resulting from the build.
    """
    buildroot_dir = _ancestor_dir(os.path.abspath(__file__), 3)
    so_dir = os.path.join(buildroot_dir, 'lib')
    so_dir_absolute = os.path.abspath(so_dir)

    return so_dir_absolute

def get_postbuild_so_pathnames():
    """
    Return a list of absolute pathnames to the shared objects that are under mxnet_lib_dir
    and should be included in a binary distribution of mxnet.

    Note that some shared objects may not actually exist depending on how the mxnet build
    was configured.
    """
    lib_filenames = [
        'libcpu_backend.so',
        'libinterpreter_backend.so',
        'libiomp5.so',
        'libmkldnn.so',
        'libmkldnn.so.0',
        'libmkldnn.so.0.17.0',
        'libmklml_intel.so',
        'libmxnet.so',
        'libngraph.so',
        'libngraph_test_util.so',
        'libnop_backend.so',
        'libtbb_debug.so',
        'libtbb_debug.so.2',
        'libtbb.so',
        'libtbb.so.2',
        ]

    so_dir = get_postbuild_so_dir()
    if not os.path.isdir(so_dir):
        raise Exception('Unable to find the post-build "lib" directory: "{}"'.format(so_dir))

    lib_paths = [ os.path.join(so_dir, l) for l in lib_filenames ]
    return lib_paths

def get_postbuild_libmxnet_pathname():
    """
    Assume that MXnet was successfully built in the source tree.  Using this file's (libinfo.py)
    location, determine the expected location of libmxnet.so, and return it as an absolute pathname.
    """
    so_dir = get_postbuild_so_dir()
    return os.path.join(so_dir, 'libmxnet.so')

def get_installed_libmxnet_pathname():
    # Assume the following installation layout:
    # ???/
    #   lib/
    #     python3.5/
    #       lib/
    #          libmxnet.so, et al
    #       site-packages/
    #          $package_name/
    #            libinfo.py (the actual instance of this file that's currently running)
    install_root = _ancestor_dir(__file__, 5)
    install_so_dir = os.path.join( install_root, 'lib', 'python3.5', 'lib' )
    install_so_pathname = os.path.abspath(os.path.join(install_so_dir, 'libmxnet.so'))
    return install_so_pathname

def get_env_libmxnet_pathname_or_None():
    lib_from_env = os.environ.get('MXNET_LIBRARY_PATH')
    if not lib_from_env:
        return None

    if not os.path.isfile(lib_from_env):
        logging.warning("MXNET_LIBRARY_PATH '{}' doesn't exist".format(lib_from_env))
        return None

    if os.path.isabs(lib_from_env):
        lib_from_env_absolute = lib_from_env
    else:
        logging.warning("MXNET_LIBRARY_PATH should be an absolute path, instead of '{}'".format(
                lib_from_env))
        lib_from_env_absolute = os.path.abspath(lib_from_env)

    # This might warrant a refactor - this function probably shouldn't have side-effects. -cconvey
    if os.name == 'nt':
        os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(lib_from_env_absolute)

    return lib_from_env_absolute

def try_find_runtime_libmxnet_pathname():
    """
    Try to find an appropriate instance of libmxnet.so to load, based on any of the following
    scenarios perhaps being true:

    (1) For reasons only the user knows, they want us to first look in the directory specified
        by the environment variable  MXNET_LIBRARY_PATH.

    (2) The user has completed an in-source build of mxnet, and is running 'import mxnet' directly
        in the source tree's 'python' directory.

    (3) The user has completed an in-source build of mxnet, and then did the following:
        'cd python; pip install -e .' and is now running 'import mxnet'.

    (4) The user installed the MXnet wheel, and is running 'import mxnet' from an arbitrary working
        directory.

    Search the expected directories according to the order shown above, and return the absolute
    pathname of the first matching copy of libmxnet.so.  If libmxnet.so isn't found in any of
    those locations, return raise an Exception containing diagnostic details.
    """
    search_paths = [
        # scenario (1)
        get_env_libmxnet_pathname_or_None(),

        # scenario (2)
        get_postbuild_libmxnet_pathname(),

        # Scenarios (3) and (4) actually present similarly.  The main difference is whether or
        # not certain files are symlinks, but that's not a problem here.
        get_installed_libmxnet_pathname(),
        ]

    for p in search_paths:
        if (p is not None) and os.path.isfile(p):
            return p

    err_msg = 'Unable to find libmxnet.so in any of these paths:'
    for p in search_paths:
        err_msg += ('\n   "{}"'.format(p))
    raise Exception(err_msg)

#def find_bundled_dynamic_lib_paths():
#    """Find MXNet dynamic library files.
#
#    with_ngraph : Iff True, the returned list should also provide the paths to any nGraph-supplied
#        shared objects that must be provided by the mxnet package.  If those files cannot be found,
#        raise a RuntimeError.
#
#    Returns
#    -------
#    lib_path : list(string)
#        List of all found path to the libraries.
#    """
#
#    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#    api_path = os.path.join(curr_path, '../../lib/')
#    cmake_build_path = os.path.join(curr_path, '../../build/')
#    dll_path = [curr_path, api_path, cmake_build_path]
#    if os.name == 'nt':
#        dll_path.append(os.path.join(curr_path, '../../build'))
#        vs_configuration = 'Release'
#        if platform.architecture()[0] == '64bit':
#            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
#            dll_path.append(os.path.join(curr_path, '../../windows/x64', vs_configuration))
#        else:
#            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
#            dll_path.append(os.path.join(curr_path, '../../windows', vs_configuration))
#    elif os.name == "posix" and os.environ.get('LD_LIBRARY_PATH', None):
#        dll_path[0:0] = [p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")]
#    if os.name == 'nt':
#        os.environ['PATH'] = os.path.dirname(__file__) + ';' + os.environ['PATH']
#        dll_path = [os.path.join(p, 'libmxnet.dll') for p in dll_path]
#    elif platform.system() == 'Darwin':
#        dll_path = [os.path.join(p, 'libmxnet.dylib') for p in dll_path] + \
#                   [os.path.join(p, 'libmxnet.so') for p in dll_path]
#    else:
#        dll_path.append('../../../')
#        dll_path = [os.path.join(p, 'libmxnet.so') for p in dll_path]
#    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
#    if len(lib_path) == 0:
#        raise RuntimeError('Cannot find the MXNet library.\n' +
#                           'List of candidates:\n' + str('\n'.join(dll_path)))
#    if os.name == 'nt':
#        os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(lib_path[0])
#
#    # We're only intersted in the entry at the head of the list...
#    lib_path = lib_path[0:1]
#
#    mxnet_lib_dir = os.path.dirname(lib_path[0])
#    lib_path += _get_ngraph_shared_lib_paths(mxnet_lib_dir)
#
#    return lib_path

# current version
#__version__ = "1.3.1"
__version__ = "1.0b0"

