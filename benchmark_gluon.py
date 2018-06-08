import mxnet as mx
import mxnet.gluon.model_zoo.vision as models
import time
import logging
import argparse

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Gluon modelzoo-based CNN perf')

parser.add_argument('--mode', type=str, default='symbolic')
parser.add_argument('--hybridized', type=bool, default=True)
parser.add_argument('--model', type=str, default='all')
parser.add_argument('--num-layer', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=0)
parser.add_argument('--type', type=str, default='inf')
parser.add_argument('--with-bn', type=bool, default=False)
parser.add_argument('--gpus', type=str, default='')

opt = parser.parse_args()

num_batches = 100
dry_run = 10  # use 10 iterations to warm up
batch_inf = [1, 2, 4, 8, 16, 32, 64, 128, 256]
batch_train = [1, 2, 4, 8, 16, 32, 64, 126, 256]
image_shapes = [(3, 224, 224), (3, 299, 299)]


def get_network(network, num_layer, with_bn=False):
    if network in ('vgg', 'resnetv1', 'resnetv2', 'densenet'):
        if num_layer == 0:
            assert "For VGG, Resnet, DenseNet, layer number must be specified!"
    if 'vgg' in network:
        network = network + str(num_layer)
    elif 'resnetv1' in network:
        network = 'resnet' + str(num_layer) +'_v1'
    elif 'resnetv2' in network:
        network = 'resnet' + str(num_layer) +'_v2'
    elif 'densenet' in network:
        network = 'densenet' + str(num_layer)
    elif 'inception-v3' in network:
        network = 'inceptionv3'
    else:
        return network
    return network

def score(network, mode, batch_size, ctx, hybridized=True):
    net = models.get_model(network)
    if 'inceptionv3' == network:
        data_shape = [('data', (batch_size,) + image_shapes[1])]
    else:
        data_shape = [('data', (batch_size,) + image_shapes[0])]

    if hybridized:
        net.hybridize()
    if mode == 'symbolic':
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=ctx)
        mod.bind(for_training     = False,
                 inputs_need_grad = False,
                 data_shapes      = data_shape)
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
        if mx.cpu() in ctx:
            data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.cpu()) for _, shape in mod.data_shapes]
        elif mx.gpu(0) in ctx:
            data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.gpu()) for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, [])
        for i in range(dry_run + num_batches):
            if i == dry_run:
                tic = time.time()
            mod.forward(batch, is_train=False)
            for output in mod.get_outputs():
                output.wait_to_read()
        fwd = time.time() - tic
    return fwd


def train(network, mode, batch_size, ctx, hybridized=True):
    net = models.get_model(network)
    if 'inceptionv3' == network:
        data_shape = [('data', (batch_size,) + image_shapes[1])]
    else:
        data_shape = [('data', (batch_size,) + image_shapes[0])]

    if hybridized:
        net.hybridize()
    if mode == 'symbolic':
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=ctx)
        mod.bind(for_training     = True,
                 inputs_need_grad = False,
                 data_shapes      = data_shape)
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
        mod.init_optimizer(kvstore='local', optimizer='sgd')
        if mx.cpu() in ctx:
            data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.cpu()) for _, shape in mod.data_shapes]
        elif mx.gpu(0) in ctx:
            data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.gpu()) for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, [])
        for i in range(dry_run + num_batches):
            if i == dry_run:
                tic = time.time()
            mod.forward(batch, is_train=True)
            for output in mod.get_outputs():
                output.wait_to_read()
            mod.backward()
            mod.update()
        bwd = time.time() - tic
    return bwd

if __name__ == '__main__':
    runtype = opt.type
    bs = opt.batch_size
    num_layer = opt.num_layer
    with_bn = opt.with_bn
    mode = opt.mode
    hybridized = opt.hybridized
    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    network = get_network(opt.model, num_layer, with_bn)
    print(network)
    if runtype == 'inf' or runtype == 'all':
        if bs != 0:
            fwd_time = score(network, mode, bs, context, hybridized)
            fps = (bs*num_batches)/fwd_time
            logging.info(network + ' inference perf for BS %d is %f img/s', bs, fps)
        else:
            for batch_size in batch_inf:
                fwd_time = score(network, mode, batch_size, context, hybridized)
                fps = (batch_size * num_batches) / fwd_time
                logging.info(network + ' inference perf for BS %d is %f img/s', batch_size, fps)
    if runtype == 'train' or runtype == 'all':
        if bs != 0:
            bwd_time = train(network, mode, bs, context, hybridized)
            fps = (bs*num_batches)/bwd_time
            logging.info(network + ' training perf for BS %d is %f img/s', bs, fps)
        else:
            for batch_size in batch_train:
                bwd_time = train(network, mode, batch_size, context, hybridized)
                fps = (batch_size * num_batches) / bwd_time
                logging.info(network + ' training perf for BS %d is %f img/s', batch_size, fps)



