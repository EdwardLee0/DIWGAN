import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from random import shuffle 
import threading 
import struct
import scipy.stats as st

class DataLoader(object):
    def __init__(self, image_path, input_image_path1, label_image_path1,input_image_path2, label_image_path2,image_size = 256,\
                 patch_size =128,  depth = 1, \
                 batch_size = 16, num_threads = 6,extension = 'bin'):
        
        #dicom file dir
        self.extension = extension
        self.image_path = image_path
        self.input_image_path1 = input_image_path1
        self.label_image_path1 = label_image_path1
        self.input_image_path2 = input_image_path2
        self.label_image_path2 = label_image_path2
               
        #image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        self.batch_size = batch_size
        #batch generator  prameters 
        self.num_threads = num_threads
        self.capacity  = 10* self.num_threads * self.batch_size
        self.min_queue = 1 * self.num_threads * self.batch_size

    def __call__(self,patient_path):
        P_input_path1, P_label_path1 = \
        glob(os.path.join(self.image_path,patient_path, self.input_image_path1, '*.' + self.extension)), \
        glob(os.path.join(self.image_path,patient_path, self.label_image_path1, '*.' + self.extension))

        P_input_path2, P_label_path2 = \
        glob(os.path.join(self.image_path,patient_path, self.input_image_path2, '*.' + self.extension)), \
        glob(os.path.join(self.image_path,patient_path, self.label_image_path2, '*.' + self.extension))
        
        
             #load images
        self.input_images1 = self.xshow(P_input_path1,self.patch_size,self.patch_size,self.depth)
        self.label_images1 = self.xshow(P_label_path1,self.patch_size,self.patch_size,self.depth)

        self.input_images2 = self.xshow(P_input_path2,self.patch_size,self.patch_size,self.depth)
        self.label_images2 = self.xshow(P_label_path2,self.patch_size,self.patch_size,self.depth)
    
    def xshow(self,path,nx,ny,nz):
        def normalize(img, max_ = 0.4034, min_ =-0.0227):
            img = (img - min_) / (max_  -  min_)
            return img
        image_append=[] 
        for s in path:  
            slices = open(s,"rb")
            image=np.zeros((nz,nx,ny),dtype=np.float32)
            for i in range(nz):
                for j in range(ny):
                    for k in range(nx): 
                      data=slices.read(4)
                      elem= struct.unpack("f", data)[0]
                      image[i][k][j] = elem
            slices.close()
            image=normalize(image)
            image=np.transpose(image, (1,2,0))
            print(np.shape(image))
            image_append.append(image)
        return np.array(image_append)
    
    
    def input_pipeline(self, sess, batch_size,image_size, patch_size, num_threads=6,depth = 1):
        queue1 = tf.train.slice_input_producer([self.input_images1, self.label_images1], shuffle=False,capacity=self.capacity)
        queue2 = tf.train.slice_input_producer([self.input_images2, self.label_images2], shuffle=False,capacity=self.capacity)
        image_batch1, label_batch1 = tf.train.batch(queue1, batch_size=batch_size, num_threads=num_threads, capacity=self.capacity, allow_smaller_final_batch=False)
        image_batch2, label_batch2 = tf.train.batch(queue2, batch_size=batch_size, num_threads=num_threads, capacity=self.capacity, allow_smaller_final_batch=False)
        self.coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, self.coord)
        return image_batch1, label_batch1,image_batch2, label_batch2,threads
    
def gauss_kernel(kernlen=21, nsig=3, channels=2):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter
	
def ParseBoolean (b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError ('Cannot parse string into boolean.')



