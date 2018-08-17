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

#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods, invalid-name, missing-docstring
"""
    This file tests that the notebooks requiring multi GPUs run without
    warning or exception.
"""
import unittest
from straight_dope_test_utils import _test_notebook
from straight_dope_test_utils import _download_straight_dope_notebooks

class StraightDopeMultiGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        assert _download_straight_dope_notebooks()

    # Chapter 7

    # TODO(vishaalk): module 'mxnet.gluon' has no attribute 'autograd'
    #def test_multiple_gpus_scratch(self):
    #    assert _test_notebook('chapter07_distributed-learning/multiple-gpus-scratch')

    def test_multiple_gpus_gluon(self):
        assert _test_notebook('chapter07_distributed-learning/multiple-gpus-gluon')

    # TODO(vishaalk): Do a dry run, and then enable.
    #def test_training_with_multiple_machines(self):
    #    assert _test_notebook('chapter07_distributed-learning/training-with-multiple-machines')

    # Chapter 8

    # TODO(vishaalk): Module skimage needs to be added to docker image.
    # def test_fine_tuning(self):
    #    assert _test_notebook('chapter08_computer-vision/fine-tuning')
