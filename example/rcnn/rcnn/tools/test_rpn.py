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

import argparse
import pprint
import mxnet as mx

from ..logger import logger
from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..core.loader import TestLoader
from ..core.tester import Predictor, generate_proposals
from ..utils.load_model import load_param


def test_rpn(network, dataset, image_set, root_path, dataset_path,
             ctx, prefix, epoch,
             vis, shuffle, thresh):
    # rpn generate proposal config
    config.TEST.HAS_RPN = True

    # print config
    pprint.pprint(config)

    # load symbol
    sym = eval('get_' + network + '_rpn_test')(num_anchors=config.NUM_ANCHORS)

    # load dataset and prepare imdb for training
    imdb = eval(dataset)(image_set, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1, shuffle=shuffle, has_rpn=True)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx)

    # infer shape
    data_shape_dict = dict(test_data.provide_data)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

    # check parameters
    for k in sym.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = None if test_data.provide_label is None else [k[0] for k in test_data.provide_label]
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start testing
    imdb_boxes = generate_proposals(predictor, test_data, imdb, vis=vis, thresh=thresh)
    imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Region Proposal Network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rpn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rpn_epoch, type=int)
    # rpn
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='rpn proposal threshold', default=0, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    ctx = mx.gpu(args.gpu)
    test_rpn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
             ctx, args.prefix, args.epoch,
             args.vis, args.shuffle, args.thresh)

if __name__ == '__main__':
    main()
