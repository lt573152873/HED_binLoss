import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy
import sys

caffe_root = '..'
sys.path.insert(0, caffe_root+'/python/')
import caffe


# Use GPU?
use_gpu = 1;
gpu_id = 0;

net_struct  = './deploy_hed.prototxt'
data_source = './data/ophtha_ma/test_cv4.lst'
data_root = '/home/lt/liuteng/my_dataset/hed/'
save_root = './results/'

    
with open(data_source) as f:
    imnames = f.readlines()

test_lst = [data_root + x.strip() for x in imnames]

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)


# load net
net = caffe.Net('./'+net_struct,'./hed_isbi_he_iter_15000.caffemodel', caffe.TEST);
    
for idx in range(0,len(test_lst)):
    print("Scoring hed for image " + data_root + imnames[idx][:-1])
    
    #Read and preprocess data
    im = Image.open(test_lst[idx])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1] #BGR
    in_ -= np.array((16.4253,33.4021,65.3908))  #ma cv4
   # in_ -= np.array((18.8213,34.1824,67.6683)) #ma cv1
    #in_ -= np.array((16.5653,33.6432,64.8960)) #ma cv3
  #  in_ -= np.array((18.0448,32.3907,64.8488)) #ma cv2
#    in_ -= np.array((17.9529,34.4556,65.9638))  #ma cv5
#    in_ -= np.array((16.3797,56.2218,116.3547)) #isbi
    in_ = in_.transpose((2,0,1))
    
    #Reshape data layer
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    
    #Score the model
    net.forward()
    fuse  = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    fsig1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    fsig2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    fsig3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    fsig4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    fsig5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
   
   #Save the results
    scipy.misc.imsave(save_root+imnames[idx][:-1][0:-4]+'_fuse.png', fuse,)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig1.tif', fsig1)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig2.tif', fsig2)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig3.tif', fsig3)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig4.tif', fsig4)
#    scipy.misc.imsave(save_root+imnames[idx][:-1]+'_sig5.tif', fsig5)
