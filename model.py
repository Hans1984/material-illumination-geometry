from config import *
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tflearn.layers.conv import global_avg_pool

def conv_layer(input_layer, sz, name='conv'):
    conv = Conv2dLayer(input_layer, 
                    act = tf.nn.relu,
                    shape = [sz[0], sz[1], sz[2], sz[3]],
                    strides = [1, 1, 1, 1],
                    padding = 'SAME',
                    name =name)
    return conv

def res_block(inputres, sz, name="res_block", is_train=False):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        out_res = conv_layer(inputres, sz, name = 'c1' )
        out_res = conv_layer(out_res, sz, name = 'c2')
        out_res.outputs = tf.nn.relu(out_res.outputs + inputres.outputs)

        return out_res
 ##residual net       
def res_net_nobn(arg_dict, input, num_blocks = 52, scope = 'resnet', is_train=False):
    with tf.variable_scope(scope):
        out_res = InputLayer(input, name='res_in')
        out_res = Conv2dLayer(out_res, act = tf.nn.relu, shape = [3, 3, 3, 64], strides = [1, 2, 2, 1], padding = 'VALID', name = 'downsample_1')
        out_res = Conv2dLayer(out_res, act = tf.nn.relu, shape = [3, 3, 64, 64], strides = [1, 2, 2, 1], padding = 'VALID', name = 'downsample_2')
        for i in range(num_blocks):
            out_res = res_block(out_res, [3, 3, 64, 64], name='res_block_/%s'%i, is_train=is_train)
        out_res = Conv2dLayer(out_res, act = tf.nn.relu, shape = [3, 3, 64, 64], strides = [1, 2, 2, 1], padding = 'VALID', name = 'downsample_3')
        out_res = Conv2dLayer(out_res, act = tf.nn.relu, shape = [3, 3, 64, 64], strides = [1, 2, 2, 1], padding = 'VALID', name = 'downsample_4')
        out_res = FlattenLayer(out_res, name='flatten_layer')
        out_res = DenseLayer(out_res, n_units=ATTRB_NUM, act=tf.nn.sigmoid, name='res_out')
        return out_res
#vgg net
def vgg_encoder(arg_dict, input, scope = 'vgg_encoder', is_train=False):
    with tf.variable_scope(scope):
        net_in = tl.layers.InputLayer(input, name='input_layer')
        conv_layers, vgg16_conv_layers = encoder(net_in)
        out = FlattenLayer(conv_layers, name='flatten_layer')
        out = DenseLayer(out, n_units=ATTRB_NUM, act=tf.nn.sigmoid, name='vgg_out')
        if is_train:
           return out, vgg16_conv_layers
        else:
           return out,vgg16_conv_layers

def encoder(input_layer):

    # Convolutional layers size 1
    network     = conv_layer(input_layer, [3, 3, 3, 64], 'encoder/h1/conv_1')
    beforepool1 = conv_layer(network, [3, 3, 64, 64], 'encoder/h1/conv_2')
    network     = pool_layer(beforepool1, 'encoder/h1/pool')

    # Convolutional layers size 2
    network     = conv_layer(network, [3, 3, 64, 128], 'encoder/h2/conv_1')
    beforepool2 = conv_layer(network, [3, 3, 128, 128], 'encoder/h2/conv_2')
    network     = pool_layer(beforepool2, 'encoder/h2/pool')

    # Convolutional layers size 3
    network     = conv_layer(network, [3, 3, 128, 256], 'encoder/h3/conv_1')
    network     = conv_layer(network, [3, 3, 256, 256], 'encoder/h3/conv_2')
    beforepool3 = conv_layer(network, [3, 3, 256, 256], 'encoder/h3/conv_3')
    network     = pool_layer(beforepool3, 'encoder/h3/pool')

    # Convolutional layers size 4
    network     = conv_layer(network, [3, 3, 256, 512], 'encoder/h4/conv_1')
    network     = conv_layer(network, [3, 3, 512, 512], 'encoder/h4/conv_2')
    beforepool4 = conv_layer(network, [3, 3, 512, 512], 'encoder/h4/conv_3')
    network     = pool_layer(beforepool4, 'encoder/h4/pool')

    # Convolutional layers size 5
    network     = conv_layer(network, [3, 3, 512, 512], 'encoder/h5/conv_1')
    network     = conv_layer(network, [3, 3, 512, 512], 'encoder/h5/conv_2')
    beforepool5 = conv_layer(network, [3, 3, 512, 512], 'encoder/h5/conv_3')
    network     = pool_layer(beforepool5, 'encoder/h5/pool')

    return network, (beforepool1.outputs, beforepool2.outputs, beforepool3.outputs, beforepool4.outputs, beforepool5.outputs)

# Max-pooling layer
def pool_layer(input_layer, str):
    network = tl.layers.PoolLayer(input_layer,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name = str)

    return network

#load vgg weights
def load_vgg_weights(network, weight_file, session):
    params = []

    if weight_file.lower().endswith('.npy'):
        npy = np.load(weight_file, encoding='latin1')
        for key, val in sorted(npy.item().items()):
            if(key[:4] == "conv"):
                print("  Loading %s" % (key))
                print("  weights with size %s " % str(val['weights'].shape))
                print("  and biases with size %s " % str(val['biases'].shape))
                params.append(val['weights'])
                params.append(val['biases'])
    else:
        print('No weights in suitable .npy format found for path ', weight_file)

    print('Assigning loaded weights..')
    tl.files.assign_params(session, params, network)

    return network