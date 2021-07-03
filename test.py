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
    
    mse_sum = 0.0
    mae_sum = 0.0
    mse_list = []
    mae_list = []
    mse_sum_each = []
    mse_sum_each_list = []
    mae_sum_each = []
    mae_sum_each_list = []
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
        y_predict_score_np = (7.0 * y_predict_score_np - 1.0)/6.0
        y_predict_score_np[y_predict_score_np<0] = 0.0
        gt_pred_np = np.squeeze(gt_pred)

        gt_pred_np_list = gt_pred_np.tolist()  
        y_predict_score_list = y_predict_score_np.tolist()
        predictor_score.append(y_predict_score_list)
        gt_score.append(gt_pred_np_list)

        # MSE loss
        loss_each, loss_mean = loss_mse_np_each(y_predict_score_list, gt_pred_np_list)
        mse_list.append(loss_mean)

        # MAE loss
        loss_each_mae, loss_mean_mae = loss_mae_np_each(y_predict_score_list, gt_pred_np_list)
        mae_list.append(loss_mean_mae)

        print('y_predict_score is :', y_predict_score_np.tolist())
        print('gt is :', gt_pred_np.tolist())
        print('loss_mse is :', loss_mean)
        print('loss_mse_each is:', loss_each)
        print('loss_mae is :', loss_mean_mae)
        print('loss_mae_each is:', loss_each_mae)

        mse_sum_each_list.append(loss_each)
        mse_sum = mse_sum + loss_mean
        mse_sum_each = np.sum([mse_sum_each, loss_each], axis = 0)

        mae_sum_each_list.append(loss_each_mae)
        mae_sum = mae_sum + loss_mean_mae
        mae_sum_each = np.sum([mae_sum_each, loss_each_mae], axis = 0)

    mse_mean = mse_sum/len(input_files)
    mse_mean_each = np.array(mse_sum_each)/len(input_files)
    print('mse_mean is :', mse_mean)
    print('mse_mean_each is :', mse_mean_each.tolist())
   
    mae_mean = mae_sum/len(input_files)
    mae_mean_each = np.array(mae_sum_each)/len(input_files)
    print('mae_mean is :', mae_mean)
    print('mae_mean_each is :', mae_mean_each.tolist())

    mse_sum_each_list = np.array(mse_sum_each_list)
    gt_score = np.array(gt_score)
    predictor_score = np.array(predictor_score)
    mae_sum_each_list = np.array(mae_sum_each_list)


##change the save_path to your own path
##the predited score with the ground truth score
    dataframe_gt = pd.DataFrame({'Name': file_name_list, 'glossiness(Pred)':predictor_score[:, 0], 'glossiness(GT)':gt_score[:, 0], 'refsharp(Pred)':predictor_score[:, 1], 'refsharp(GT)':gt_score[:, 1], 'contgloss(Pred)':predictor_score[:, 2], 'contgloss(GT)':gt_score[:, 2], 'metallicness(Pred)':predictor_score[:, 3], 'metallicness(GT)':gt_score[:, 3], 'lightness(Pred)':predictor_score[:, 4], 'lightness(GT)':gt_score[:, 4], 'anisotropy(Pred)':predictor_score[:, 5], 'anisotropy(GT)':gt_score[:, 5]})
    dataframe_gt.to_csv("./results/predicted_score.csv", index=False,sep=',')

    sess.close()


