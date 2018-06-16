
import numpy as np


import numpy
path = 'savedweights\\'


def save_conv_variable(conv_w,filename_w,conv_b,filename_b):
    width = conv_w.shape[0]
    height = conv_w.shape[1]
    w = conv_w[:,:,0,0]
    for i in range(0,conv_w.shape[3]):
        for j in range(0,conv_w.shape[2]):
            w = numpy.append(w,conv_w[:,:,j,i])
    w = w[width*height:]
    numpy.savetxt(filename_w, w, delimiter=',', newline=',\n')

    numpy.savetxt(filename_b, conv_b, delimiter=',', newline=',\n')

def save_fc_variable(fc_w,filename_w,fc_b,filename_b):
    numpy.savetxt(filename_w,fc_w,delimiter=',',newline=',\n')
    numpy.savetxt(filename_b,fc_b,delimiter=',',newline=',\n')

def save_reshape_fc_variable(reshape_dim,fc_w,filename_w,fc_b,filename_b):
    features = reshape_dim[2]
    row = reshape_dim[0]
    col = reshape_dim[1]
    a = np.zeros([row*col*features,fc_w.shape[1]],dtype=float)
    t = 0
    for k in range(features):
        for j in range(row):
            for i in range(col):
                a[t, :] = fc_w[i * features + j * col * features + k, :]
                t = t + 1
    numpy.savetxt(filename_w,a,delimiter=',',newline=',\n')
    numpy.savetxt(filename_b,fc_b,delimiter=',',newline=',\n')

def save_bn_variable(moving_mean,moving_variance,beta=0,gamma=1,filename='bn'):
    numpy.savetxt(filename+'_beta.txt',beta,delimiter=',',newline=',\n')
    numpy.savetxt(filename + '_gamma.txt', gamma, delimiter=',', newline=',\n')
    numpy.savetxt(filename + '_moving_mean.txt', moving_mean, delimiter=',', newline=',\n')
    numpy.savetxt(filename + '_moving_variance.txt', moving_variance, delimiter=',', newline=',\n')
