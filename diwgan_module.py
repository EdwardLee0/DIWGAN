import os
import tensorflow as tf
import numpy as np

def generator(image1,image2, name="generator",padding='same',reuse = True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        conv11 = tf.layers.conv2d(image1, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv11', use_bias=False)
        conv11 = tf.nn.relu(conv11)
        conv21 = tf.layers.conv2d(conv11, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv21', use_bias=False)
        conv21 = tf.nn.relu(conv21)
        # 128
        conv1 = tf.layers.conv2d(image2, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
        conv1= tf.nn.relu(conv1)
        conv2 = tf.layers.conv2d(conv1, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
        conv2 = tf.nn.relu(conv2)

        pool11 = tf.layers.max_pooling2d(inputs=conv11, pool_size=2, strides=2,padding=padding,name = 'pool11')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2,padding=padding,name = 'pool1')

        conv31 = tf.layers.conv2d(pool11, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv31', use_bias=False)
        conv31 = tf.nn.relu(conv31)
        conv41 = tf.layers.conv2d(conv31, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv41', use_bias=False)
        conv41 = tf.nn.relu(conv41)
        # 64
        conv3 = tf.layers.conv2d(pool1, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv2d(conv3, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
        conv4 = tf.nn.relu(conv4)

        pool21 = tf.layers.max_pooling2d(inputs=conv41, pool_size=2,strides=2,padding=padding,name = 'pool21')
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2,strides=2,padding=padding,name = 'pool2')

        conv51 = tf.layers.conv2d(pool21, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv51', use_bias=False)
        conv51 = tf.nn.relu(conv51)
        conv61 = tf.layers.conv2d(conv51, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv61', use_bias=False)
        conv61 = tf.nn.relu(conv61)
        # 32
        conv5 = tf.layers.conv2d(pool2, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5', use_bias=False)
        conv5 = tf.nn.relu(conv5)
        conv6 = tf.layers.conv2d(conv5, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6', use_bias=False)
        conv6 = tf.nn.relu(conv6)

        pool31 = tf.layers.max_pooling2d(inputs=conv61, pool_size=2,strides=2,padding=padding,name = 'pool31')
        pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=2,strides=2,padding=padding,name = 'pool3')


        conv71 = tf.layers.conv2d(pool31, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv71', use_bias=False)
        conv71 = tf.nn.relu(conv71)
        conv81 = tf.layers.conv2d(conv71, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv81', use_bias=False)
        conv81 = tf.nn.relu(conv81)
        # 16
        conv7 = tf.layers.conv2d(pool3, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv7', use_bias=False)
        conv7 = tf.nn.relu(conv7)
        conv8 = tf.layers.conv2d(conv7, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv8', use_bias=False)
        conv8 = tf.nn.relu(conv8)

        up11 = tf.layers.conv2d_transpose(conv81,256,3,strides=2, padding=padding,name='decon11')
        up11 = tf.concat([up11, conv61, conv6], axis=-1, name='concat11')
        # 32
        up1 = tf.layers.conv2d_transpose(conv8,256,3,strides=2, padding=padding,name='decon1')
        up1 = tf.concat([up1, conv6,conv61], axis=-1, name='concat1')

        uconv11 = tf.layers.conv2d(up11, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv11', use_bias=False)
        uconv11 = tf.nn.relu(uconv11)
        uconv21 = tf.layers.conv2d(uconv11, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv21', use_bias=False)
        uconv21 = tf.nn.relu(uconv21)

        uconv1 = tf.layers.conv2d(up1, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv1', use_bias=False)
        uconv1 = tf.nn.relu(uconv1)
        uconv2 = tf.layers.conv2d(uconv1, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv2', use_bias=False)
        uconv2 = tf.nn.relu(uconv2)

        up21 = tf.layers.conv2d_transpose(uconv21,128,3,strides=2, padding=padding,name='decon21')
        up21 = tf.concat([up21, conv41,conv4], axis=-1, name='concat21')
        # 64
        up2 = tf.layers.conv2d_transpose(uconv2,128,3,strides=2, padding=padding,name='decon2')
        up2 = tf.concat([up2, conv4, conv41], axis=-1, name='concat2')

        uconv31 = tf.layers.conv2d(up21, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv31', use_bias=False)
        uconv31 = tf.nn.relu(uconv31)
        uconv41 = tf.layers.conv2d(uconv31, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv41', use_bias=False)
        uconv41 = tf.nn.relu(uconv41)

        uconv3 = tf.layers.conv2d(up2, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv3', use_bias=False)
        uconv3 = tf.nn.relu(uconv3)
        uconv4 = tf.layers.conv2d(uconv3, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv4', use_bias=False)
        uconv4 = tf.nn.relu(uconv4)


        up31 = tf.layers.conv2d_transpose(uconv41, 64 , 3 ,strides=2, padding=padding, name='decon31')
        up31 = tf.concat([up31,conv21,conv2], axis=-1, name='concat31')
        # 128
        up3 = tf.layers.conv2d_transpose(uconv4, 64 , 3 ,strides=2, padding=padding, name='decon3')
        up3 = tf.concat([up3,conv2,conv21], axis=-1, name='concat3')


        uconv51 = tf.layers.conv2d(up31, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv51', use_bias=False)
        uconv51 = tf.nn.relu(uconv51)
        uconv61 = tf.layers.conv2d(uconv51, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv61', use_bias=False)
        uconv61 = tf.nn.relu(uconv61)

        uconv5 = tf.layers.conv2d(up3, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv5', use_bias=False)
        uconv5 = tf.nn.relu(uconv5)
        uconv6 = tf.layers.conv2d(uconv5, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='uconv6', use_bias=False)
        uconv6 = tf.nn.relu(uconv6)


        output1 = tf.layers.conv2d(uconv6, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_out', use_bias=False)
        output1 = tf.nn.relu(output1)

        output = tf.layers.conv2d(uconv61, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_out1', use_bias=False)
        output = tf.nn.relu(output)
        return output1,output 


def discriminator1(image, name="discriminator1",reuse = True):
     with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l11 = tf.layers.conv2d(image, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv11')
        l11 = tf.nn.relu(l11)

        l21 = tf.layers.conv2d(l11, 64, 3, padding='same',  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv21')
        l21 = tf.nn.relu(l21)

        l31 = tf.layers.conv2d(l21, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv31')
        l31 = tf.nn.relu(l31)

        l41 = tf.layers.conv2d(l31, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv41')
        l41 = tf.nn.relu(l41)

        l51 = tf.layers.conv2d(l41, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv51')
        l51 = tf.nn.relu(l51)

        l61 = tf.layers.conv2d(l51, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv61')
        l61 = tf.nn.relu(l61)

        
        fc11 = tf.contrib.layers.flatten(l61)
        fc11 = tf.layers.dense(fc11, units=1024, name='dense11')
        fc11 = leaky_relu(fc11, alpha=0.2)
        fc21 = tf.layers.dense(fc11, units=1, name='dense21')
        
        return fc21



def discriminator2(image, name="discriminator2",reuse = True):
     with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l1 = tf.layers.conv2d(image, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv1')
        l1 = tf.nn.relu(l1)

        l2 = tf.layers.conv2d(l1, 64, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv2')
        l2 = tf.nn.relu(l2)

        l3 = tf.layers.conv2d(l2, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv3')
        l3 = tf.nn.relu(l3)

        l4 = tf.layers.conv2d(l3, 128, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv4')
        l4 = tf.nn.relu(l4)

        l5 = tf.layers.conv2d(l4, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv5')
        l5 = tf.nn.relu(l5)

        l6 = tf.layers.conv2d(l5, 256, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='d_conv6')
        l6 = tf.nn.relu(l6)

        
        fc1 = tf.contrib.layers.flatten(l6)
        fc1 = tf.layers.dense(fc1, units=1024, name='dense1')
        fc1 = leaky_relu(fc1, alpha=0.2)
        fc2 = tf.layers.dense(fc1, units=1, name='dense2')
        
        return fc2




def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")



def leaky_relu(inputs, alpha):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)

def gdl_loss(gen_CT, gt_CT, alpha, batch_size_tf):
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0) 
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)]) 
    strides = [1, 1, 1, 1]  
    padding = 'SAME'
    gen_dx = tf.abs(tf.nn.conv2d(gen_CT, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_CT, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_CT, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_CT, filter_y, strides, padding=padding))
    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)
    gdl=tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha))/tf.cast(batch_size_tf,tf.float32)
    return gdl

