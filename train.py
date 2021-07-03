import os, time
import numpy as np
import tensorflow as tf
from model import *
from utils import *
from easydict import EasyDict as edict

def train(arg_dict):
    log_dir = arg_dict.log_dir + '/train/scalar'
    ckpt_dir = arg_dict.ckpt_dir
    tl.files.exists_or_mkdir(log_dir)
    tl.files.exists_or_mkdir(ckpt_dir)
    global_step = tf.Variable(0, trainable=False)

    # Load data
    input_files, gt_values = load_data_cvs_exclude(arg_dict, True)
    print('the lenght is :', len(gt_values))
    input_files = np.array(input_files)
    gt_values = np.array(gt_values)
    length_data = len(gt_values)

    ## Define session
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

    ## Define the network ##
    with tf.variable_scope('input'):
        input_imgs_ph = tf.placeholder('float32', [arg_dict.batch_size, arg_dict.patch_size, arg_dict.patch_size, 3], name = 'input_image')
        label_pred_ph = tf.placeholder('float32', [arg_dict.batch_size, ATTRB_NUM], name = 'ground_truth')
    with tf.variable_scope('vgg_net') as scope:
        output_pred, _ = vgg_encoder(arg_dict, input_imgs_ph,  scope=scope, is_train=True)
        y = output_pred.outputs
    ## Define loss ##
    with tf.variable_scope('loss'):
        with tf.variable_scope('mse'):
            loss = tl.cost.mean_squared_error(y, label_pred_ph, is_mean = True, name = 'mse_loss')
        loss_total = tf.identity(loss, name='total')

    ## Define optomizer ##
    train_params = output_pred.all_params
    
    with tf.variable_scope('Optimizer'):
        lr_starter = tf.Variable(arg_dict.lr, trainable = False)
        steps_per_epoch = length_data/config.TRAIN.batch_size
        lr = tf.train.exponential_decay(lr_starter, global_step,
                                           int(steps_per_epoch//5), 0.99, staircase=False)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optim = tf.train.AdamOptimizer(lr, beta1 = arg_dict.beta_1).minimize(loss_total, global_step, var_list = train_params)

    ## Define summary ##
    writer_scalar = tf.summary.FileWriter(log_dir, sess.graph, flush_secs=30, filename_suffix = '.loss_log')
    loss_list =[]
    loss_list.append(tf.summary.scalar('total_loss', loss_total))
    loss_sum = tf.summary.merge(loss_list)

    ## Start training =========================================================================================================
    
    ## load the pre-trained parameters
    sess.run(tf.global_variables_initializer())
    if(arg_dict.re_train):
        # Load model weights
        print("\n\nLoading trained parameters from '%s'..." % arg_dict.pretrained_model)
        load_params = tl.files.load_npz(name=arg_dict.pretrained_model)
        tl.files.assign_params(sess, load_params, output_pred)
        print("...done!\n")
    elif(arg_dict.pre_vgg):
        # Load pretrained VGG16 weights for encoder
        print("\n\nLoading parameters for VGG16 convolutional layers, from '%s'..." % arg_dict.vgg16_parameters)
        load_vgg_weights(vgg16_conv_layers, arg_dict.vgg16_parameters, sess)
        print("...done!\n")

    step = 0
    for ep in range(arg_dict.epoch):
        n_iter = 0
        sf_idx = np.arange(len(input_files))
        print('The length of training dataset!')
        print(len(input_files))
        np.random.shuffle(sf_idx)
        input_files = input_files[sf_idx]
        gt_values = gt_values[sf_idx]

        ep_time = time.time()
        for idx in range(0, len(input_files), arg_dict.batch_size):
            step_time = time.time()
            b_idx = (idx + np.arange(arg_dict.batch_size)) % len(input_files)

            input_imgs = read_imgs_augmentation(input_files[b_idx])
            gt_pred = gt_values[b_idx]

            feed_dict = {input_imgs_ph: input_imgs, label_pred_ph: gt_pred}
            _ = sess.run(optim, feed_dict)

            err_total, lr_val = sess.run([loss_total, lr], feed_dict)

            print('Epoch [%2d/%2d] %4d/%4d time: %4.2fs, err[total_loss: %1.2e], lr: %1.2e' % \
                (ep, arg_dict.epoch, n_iter, len(input_files)/arg_dict.batch_size, time.time() - step_time, err_total, lr_val))

            err_sum= sess.run(loss_sum, feed_dict)
            writer_scalar.add_summary(err_sum, step)
            n_iter += 1
            step += 1
            if step % 10 == 0:
               tl.files.save_npz(output_pred.all_params, name = ckpt_dir + '/epoch_{}.npz'.format(step), sess = sess)