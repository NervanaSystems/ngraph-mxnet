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
import os.path
import platform
import logging

def _get_ngraph_shared_lib_paths(mxnet_lib_dir):
    """
    Find any shared-object files that are supplied by '3rdparty/ngraph' *and* must be provided by
    the same Python package as 'libmxnet.so'.

    If even one of the expected libraries is present, assume that MXnet was build with nGraph
    support.  Otherwise just return an empty list.

    Returns
    -------
    lib_paths : list(string)
        List of all found path to the libraries.
    """
    # Here are the needed SONAME strings, as discovered by manually running
    # `readelf -d foo.so | grep NEEDED` on each of the shared objects 'foo.so*' provided by nGraph,
    # and on `libmxnet.so`.
    #
    # We're only considering those SONAMEs that we actually want to include in the mxnet package.
    # There are other SONAMEs such as 'libstdc++.so.6' that we assume shall be supplied by the
    # operating system, etc.
    NEEDED_SONAME_strings = set([
        'libcodegen.so.0.5',
        'libcpu_backend.so.0.5',
        'libiomp5.so',
        'libmkldnn.so.0',
        'libmklml_intel.so',
        'libngraph.so.0.5',
        'libtbb.so.2',
        ])

    # Note that some of these SONAMEs correspond to symlinks to regular files.
    # E.g., 'libcpu_backend.so.0.5' -> 'libcpu_backend.so.0.5.0+ad12723'.
    # In the mxnet package we want a regular file whose name is that of the symlink
    # and whose content is that of the pointed-to regular file.
    #
    # We're relying on Python setuputils to exhibit this behavior when provided with a symlink.
    lib_paths = []
    missing_lib_paths = []
    for soname in NEEDED_SONAME_strings:
        pathname = os.path.join(mxnet_lib_dir, soname)
        if os.path.isfile(pathname):
            lib_paths.append(pathname)
        else:
            missing_lib_paths.append(pathname)

    if (len(lib_paths) > 0) and (len(lib_paths) != len(NEEDED_SONAME_strings)):
        raise RuntimeError(
            'Found only some of the expected nGraph-supplied libraries.  Missing:\n'.format(
                '\n   '.join(missing_lib_paths)))

    return lib_paths


def find_lib_path():
    """Find MXNet dynamic library files.

    with_ngraph : Iff True, the returned list should also provide the paths to any nGraph-supplied
        shared objects that must be provided by the mxnet package.  If those files cannot be found,
        raise a RuntimeError.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries.
    """
    lib_from_env = os.environ.get('MXNET_LIBRARY_PATH')
    if lib_from_env:
        if os.path.isfile(lib_from_env):
            if not os.path.isabs(lib_from_env):
                logging.warning("MXNET_LIBRARY_PATH should be an absolute path, instead of: %s",
                                lib_from_env)
            else:
                if os.name == 'nt':
                    os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(lib_from_env)
                return [lib_from_env]
        else:
            logging.warning("MXNET_LIBRARY_PATH '%s' doesn't exist", lib_from_env)

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = os.path.join(curr_path, '../../lib/')
    cmake_build_path = os.path.join(curr_path, '../../build/')
    dll_path = [curr_path, api_path, cmake_build_path]
    if os.name == 'nt':
        dll_path.append(os.path.join(curr_path, '../../build'))
        vs_configuration = 'Release'
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
            dll_path.append(os.path.join(curr_path, '../../windows/x64', vs_configuration))
        else:
            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
            dll_path.append(os.path.join(curr_path, '../../windows', vs_configuration))
    elif os.name == "posix" and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path[0:0] = [p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")]
    if os.name == 'nt':
        os.environ['PATH'] = os.path.dirname(__file__) + ';' + os.environ['PATH']
        dll_path = [os.path.join(p, 'libmxnet.dll') for p in dll_path]
    elif platform.system() == 'Darwin':
        dll_path = [os.path.join(p, 'libmxnet.dylib') for p in dll_path] + \
                   [os.path.join(p, 'libmxnet.so') for p in dll_path]
    else:
        dll_path.append('../../../')
        dll_path = [os.path.join(p, 'libmxnet.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find the MXNet library.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    if os.name == 'nt':
        os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(lib_path[0])

    # We're only intersted in the entry at the head of the list...
    lib_path = lib_path[0:1]

    mxnet_lib_dir = os.path.dirname(lib_path[0])

    # TODO:
    # This is only needed when we add support for building Pip Wheels enabled
    # with nGraph.  For now we disable this because it breaks non-nGraph
    # builds.
    #lib_path += _get_ngraph_shared_lib_paths(mxnet_lib_dir)

    return lib_path


# current version
__version__ = "1.3.1"
