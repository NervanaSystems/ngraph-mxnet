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

"""Train CIFAR-10 classifier in MXNet.
Demonstrates using the Module class.
"""
import logging
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image-classification")))

import find_mxnet
import mxnet as mx
import argparse
import train_model
import importlib
import platform


def command_line_args(defaults=False):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--network', type=str, default='inception-bn-28-small',
                        help = 'which CNN style to use')
    my_dir = os.path.dirname(__file__)
    default_data_dir = os.path.abspath(os.path.join(my_dir, '..', 'image-classification', 'cifar10')) + '/'
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.05,
                        help='the initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    if defaults:
        return parser.parse_args([])
    else:
        return parser.parse_args()


# download data if necessary
def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    cwd = os.path.abspath(os.getcwd())
    os.chdir(data_dir)
    if (not os.path.exists('train.rec')) or \
       (not os.path.exists('test.rec')) :
        import urllib, zipfile, glob
        dirname = os.getcwd()
        zippath = os.path.join(dirname, "cifar10.zip")
        urllib.urlretrieve("http://data.mxnet.io/mxnet/data/cifar10.zip", zippath)
        zf = zipfile.ZipFile(zippath, "r")
        zf.extractall()
        zf.close()
        os.remove(zippath)
        for f in glob.glob(os.path.join(dirname, "cifar", "*")):
            name = f.split(os.path.sep)[-1]
            os.rename(f, os.path.join(dirname, name))
        os.rmdir(os.path.join(dirname, "cifar"))
    os.chdir(cwd)

# data
def get_iterator(args, kv):
    data_shape = (3, 28, 28)
    data_dir = args.data_dir
    if os.name == "nt":
        data_dir = data_dir[:-1] + "\\"
    if '://' not in args.data_dir:
        _download(data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = data_dir + "train.rec",
        mean_img    = data_dir + "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = data_dir + "test.rec",
        mean_img    = data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)


def do_train(args, callback_args=None):
    # network
    net = importlib.import_module('symbol_' + args.network).get_symbol(10)

    my_dir = os.path.dirname(__file__)
    if args.model_prefix is not None:
        args.model_prefix = os.path.abspath(os.path.join(my_dir, args.model_prefix))
    if args.save_model_prefix is not None:
        args.save_model_prefix = os.path.abspath(os.path.join(my_dir, args.save_model_prefix))


    ################################################################################
    # train
    ################################################################################

    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    logging.info('running on %s', platform.node())

    (train, val) = get_iterator(args, kv)

    if args.gpus is None or args.gpus == '':
        devs = mx.cpu()
    elif type(args.gpus) == str:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        devs = mx.gpu(int(args.gpus))
    logging.info('Starting with devices %s', devs)

    mod = mx.mod.Module(net, context=devs)

    # load model
    model_prefix = args.model_prefix

    if args.load_epoch is not None:
        assert model_prefix is not None
        logging.info('loading model from %s-%d...' % (model_prefix, args.load_epoch))
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    optim_args = {'learning_rate': args.lr, 'wd': 0.00001, 'momentum': 0.9}
    if 'lr_factor' in args and args.lr_factor < 1:
        optim_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        optim_args['clip_gradient'] = args.clip_gradient

    eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

    if args.load_epoch:
        begin_epoch = args.load_epoch+1
    else:
        begin_epoch = 0

    if not callback_args:
        callback_args = {
            'batch_end_callback': mx.callback.Speedometer(args.batch_size, 50),
            'epoch_end_callback': checkpoint,
        }
    else:
        pass
        #TODO: add checkpoint back in

    logging.info('start training for %d epochs...', args.num_epochs)
    mod.fit(train, eval_data=val, optimizer_params=optim_args,
            eval_metric=eval_metrics, num_epoch=args.num_epochs,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch,
            **callback_args)

if __name__ == "__main__":
    args = command_line_args()
    do_train(args)

