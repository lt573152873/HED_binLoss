import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy
import sys

caffe_root = '../../'
sys.path.insert(0, caffe_root+'/python/')
import caffe


# Use GPU?
use_gpu = 1;
gpu_id = 0;

#net_struct  = 'deploy.prototxt'
net_struct  = 'deploy_sdsn.prototxt'
data_source = '../../data/HED-BSDS/test.lst'

data_root = '../../data/HED-BSDS/'
save_root = './results/'

    
with open(data_source) as f:
    imnames = f.readlines()

test_lst = [data_root + x.strip() for x in imnames]

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)


# load net
#net = caffe.Net('./'+net_struct,'/home/guosong/dataset/caffemodel/hed_pretrained_bsds.caffemodel', caffe.TEST);
net = caffe.Net('./'+net_struct,'./hed_iter_42000.caffemodel', caffe.TEST);
	
#for idx in range(0,len(test_lst)):
for idx in range(0,20):
    print("Scoring hed for image " + data_root + imnames[idx][:-1])
    
    #Read and preprocess data
    im = Image.open(test_lst[idx])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1] #BGR
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    
    #Reshape data layer
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    
    #Score the model
    net.forward()
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    fuse  = net.blobs['sigmoid-fuse'].data[0][0,:,:]
   
   #Save the results
    fuse = 255*(1 - fuse)
    out1 = 255*(1-out1)
    out2 = 255*(1-out2)
    out3 = 255*(1-out3)
    out4 = 255*(1-out4)
    out5 = 255*(1-out5)
    scipy.misc.imsave(save_root+imnames[idx][:-1][0:-3]+'_fuse.png', fuse,)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig1.tif', out1)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig2.tif', out2)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig3.tif', out3)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig4.tif', out4)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig5.tif', out5)
