from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import math
import numpy as np
def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x))
    return y
def ReLu(x):
    return np.maximum(x,0)
def Conv2D(x, W, bias,strides=[1,1,1,1], padding='SAME'):
    if len(x.shape)!=4:
        print("input shape must be [samples,width,height,channel!]")
        return -1

    pad_row = W.shape[0]-1
    pad_col = W.shape[1]-1
    filter_num = W.shape[3]

    sample_num = x.shape[0]
    width = x.shape[1]
    height = x.shape[2]
    channel = x.shape[3]

    if padding=='SAME':
        new_width = width+pad_row
        new_height = height+pad_col

        x_pad = np.zeros((sample_num,new_width,new_height,channel),dtype=float)
        x_pad[:,int(pad_row/2):new_width-int(pad_row/2),int(pad_col/2):new_height-int(pad_col/2),:] = x
        x_output = np.zeros((sample_num,width,height,filter_num), dtype=float)

        for sample in range(sample_num):
            for fi in range(filter_num):
                for i in range(width):
                    for j in range(height):
                        x_output[sample, i, j, fi] = x_output[sample,i,j,fi] + bias[fi]
                        for ch in range(channel):
                            mul = np.multiply(x_pad[sample,i:i+W.shape[0],j:j+W.shape[1],ch],W[:,:,ch,fi])
                            x_output[sample,i,j,fi] = x_output[sample,i,j,fi]+np.sum(np.sum(mul))
    if padding=='VALID':
        output_width = math.ceil(
            (width - W.shape[0] + 1) /
            strides[1])
        output_height = math.ceil(
            (height - W.shape[1] + 1) /
            strides[2])
        x_output = np.zeros((sample_num,output_width,output_height,filter_num),dtype=np.float32)

        for sample in range(sample_num):
            for fi in range(filter_num):
                for i in range(output_width):
                    for j in range(output_height):
                        x_output[sample, i, j, fi] = x_output[sample,i,j,fi] + bias[fi]
                        for ch in range(channel):
                            ii = i*strides[1]
                            jj = j*strides[2]
                            mul = np.multiply(x[sample,ii:ii+W.shape[0],jj:jj+W.shape[1],ch],W[:,:,ch,fi])
                            x_output[sample,i,j,fi] = x_output[sample,i,j,fi]+np.sum(np.sum(mul))

    for i in range(0,31,3):
        print(i)




    return x_output

def max_pool_2x2(x):
    width = x.shape[1]
    height = x.shape[2]
    m=0;
    n=0;
    x_out = np.zeros((x.shape[0], int(x.shape[1]/2), int(x.shape[2]/2), x.shape[3]), dtype=float)
    for filter_index in range(x.shape[3]):
        m=0
        for i in range(0,width,2):
            n=0;
            for j in range(0,height,2):
                x_out[0,m,n,filter_index] = np.max(x[0,i:i+2,j:j+2,filter_index])
                n=n+1
            m=m+1
    return x_out
def fc(x,W,bias):
    x_reshape = x.reshape(x.shape[0],-1)
    return np.dot(x_reshape,W)+bias

def bn(x,moving_mean,moving_variance,gamma=1,beta=0,center=False,scale=False):
    variance_epsilon = 0.0000001
    if center==False:
        beta = 0
    if scale==False:
        gamma=1
    y = beta + gamma * (x - moving_mean) / np.sqrt(
        moving_variance + variance_epsilon)
    return y
