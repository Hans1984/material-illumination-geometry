# This file should contain the testing procedures
import os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from model import *
from utils import *
from easydict import EasyDict as edict
import scipy.misc
import pandas as pd

def test(arg_dict):
    ckpt_dir = arg_dict.ckpt_dir
    parameters_dir = ckpt_dir + '/weights.npz'
    tl.files.exists_or_mkdir(ckpt_dir)

    #load data
    input_files, gt_values = load_data_cvs_exclude(arg_dict, False)
    gt_values = np.array(gt_values)

    ## Define session
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, gpu_options=gpu_options))
    sess.run(init_op)

    ## Define the network ##
    with tf.variable_scope('input'):
        input_imgs_ph = tf.placeholder('float32', [None, arg_dict.patch_size, arg_dict.patch_size, 3], name = 'input_image')
        label_pred_ph = tf.placeholder('float32', [1, ATTRB_NUM], name = 'ground_truth')
    with tf.variable_scope('vgg_net') as scope:
        output_pred, _ = vgg_encoder(arg_dict, input_imgs_ph,  scope=scope)

    y_predict = output_pred.outputs

    # Loading model weights
    print("\n\nLoading trained parameters from '%s'..." % parameters_dir)
    load_params = tl.files.load_npz(name = parameters_dir)
    print('the parameters is :', output_pred.all_params)
    tl.files.assign_params(sess, load_params, output_pred)
    print("...done!\n")
    
    file_name_list = []
    predictor_score = []
    gt_score = []
    # Predicting
    for idx in range(len(input_files)):
        print("idx %d: '%s'"%(idx,input_files[idx]))

        file_name_list.append(input_files[idx])

        #read images
        input_imgs = read_imgs_test(input_files[idx])
        
        gt_pred = gt_values[idx]
        gt_pred = gt_pred.reshape(-1, 6)

        feed_dict = {input_imgs_ph: input_imgs, label_pred_ph: gt_pred}
        
        # convey tensor to numpy
        y_predict_score  = sess.run([y_predict], feed_dict=feed_dict)
        y_predict_score_np = np.squeeze(y_predict_score)
        gt_pred_np = np.squeeze(gt_pred)

        gt_pred_np_list = gt_pred_np.tolist()  
        y_predict_score_list = y_predict_score_np.tolist()
        predictor_score.append(y_predict_score_list)
        gt_score.append(gt_pred_np_list)

    gt_score = np.array(gt_score)
    predictor_score = np.array(predictor_score)

##change the save_path to your own path
##the predited score with the ground truth score
    dataframe_gt = pd.DataFrame({'Name': file_name_list, 'glossiness(Pred)':predictor_score[:, 0], 'glossiness(GT)':gt_score[:, 0], 'refsharp(Pred)':predictor_score[:, 1], 'refsharp(GT)':gt_score[:, 1], 'contgloss(Pred)':predictor_score[:, 2], 'contgloss(GT)':gt_score[:, 2], 'metallicness(Pred)':predictor_score[:, 3], 'metallicness(GT)':gt_score[:, 3], 'lightness(Pred)':predictor_score[:, 4], 'lightness(GT)':gt_score[:, 4], 'anisotropy(Pred)':predictor_score[:, 5], 'anisotropy(GT)':gt_score[:, 5]})
    dataframe_gt.to_csv("./results/predicted_score.csv", index=False,sep=',')

    sess.close()


