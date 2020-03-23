import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import time
from glob import glob
import inout_util as ut
import diwgan_module as modules

class diwgan(object):
    def __init__(self, sess, args):
        self.sess = sess  

        self.train_image = args.train_image
        self.test_image = args.test_image 

        self.p_info = '_'.join(self.test_image)
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = os.path.join('.', 'logs',  self.p_info)
        print('directory check!!\ncheckpoint : {}\ntensorboard_logs : {}'.format(self.checkpoint_dir, self.log_dir))

        self.g_net = modules.generator
        self.d_net1 = modules.discriminator1
        self.d_net2 = modules.discriminator2
        
        print('data load...') 
        if args.phase == 'train':
          self.image_loader = ut.DataLoader(args.image_path,args.input_path1, args.label_path1,args.input_path2, args.label_path2,\
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,\
             batch_size = args.batch_size)
        else:                             
          self.test_image_loader = ut.DataLoader(args.image_path,args.input_path1, args.label_path1,args.input_path2, args.label_path2,\
             image_size = args.whole_size, patch_size = args.whole_size, depth = args.img_channel,\
             batch_size = args.batch_size)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_image)
            print('data load complete !!!')
            self.z_i1 = tf.placeholder(tf.float32, [args.batch_size, args.patch_size, args.patch_size, args.img_channel],name='train_input1')
            self.x_i1 = tf.placeholder(tf.float32, [args.batch_size, args.patch_size, args.patch_size, args.img_channel],name='train_label1')
            self.z_i2 = tf.placeholder(tf.float32, [args.batch_size, args.patch_size, args.patch_size, args.img_channel],name='train_input2')
            self.x_i2 = tf.placeholder(tf.float32, [args.batch_size, args.patch_size, args.patch_size, args.img_channel],name='train_label2')
        else:
            self.test_image_loader(self.test_image)
            print('data load complete !!!')
            self.z_i1 = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'INPUT1')
            self.x_i1 = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'LABEL1')
            self.z_i2 = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'INPUT2')
            self.x_i2 = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'LABEL2')
        
        self.whole_z1 = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_INPUT1')
        self.whole_x1 = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_LABEL1')
        self.whole_z2 = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_INPUT2')
        self.whole_x2 = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_LABEL2')
        
        self.G_zi1,self.G_zi2 = self.g_net(self.z_i1,self.z_i2,padding='same')
        self.G_whole_zi1,self.G_whole_zi2 = self.g_net(self.whole_z1,self.whole_z2,padding='same')
        
        self.D_xi1 = self.d_net1(self.x_i1,reuse =False)
        self.D_G_zi1= self.d_net1(self.G_zi1)

        self.D_xi2 = self.d_net2(self.x_i2,reuse =False)
        self.D_G_zi2= self.d_net2(self.G_zi2)
        
        self.gdl_loss1 = modules.gdl_loss(self.G_zi1,self.x_i1,2,args.batch_size)
        self.gdl_loss2 = modules.gdl_loss(self.G_zi2,self.x_i2,2,args.batch_size)
        
        self.epsilon1 = tf.random_uniform(shape=[args.batch_size,1], minval=0.,maxval=1.)
        self.x_hat1 =self.epsilon1*tf.reshape(self.x_i1, [args.batch_size, -1]) + (1-self.epsilon1)*tf.reshape(self.G_zi1, [args.batch_size, -1])
        self.x_hat1=tf.reshape(self.x_hat1, [args.batch_size, args.patch_size, args.patch_size, args.img_channel])
        self.D_x_hat1 = self.d_net1(self.x_hat1)
        self.grad_x_hat1 = tf.gradients(self.D_x_hat1, self.x_hat1)[0]
        self.grad_x_hat_l2_1 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat1), axis=1))
        self.gradient_penalty1 =  tf.square(self.grad_x_hat_l2_1 - 1.0)

        self. w_distance1 = tf.reduce_mean(self.D_G_zi1) - tf.reduce_mean(self.D_xi1) 
        grad_penal1 =  args.lambda_g1 *tf.reduce_mean(self.gradient_penalty1 )
        self.D_loss1 = self. w_distance1 + grad_penal1


        self.epsilon2 = tf.random_uniform(shape=[args.batch_size,1], minval=0.,maxval=1.)
        self.x_hat2 =self.epsilon2*tf.reshape(self.x_i2, [args.batch_size, -1]) + (1-self.epsilon2)*tf.reshape(self.G_zi2, [args.batch_size, -1])
        self.x_hat2=tf.reshape(self.x_hat2, [args.batch_size, args.patch_size, args.patch_size, args.img_channel])
        self.D_x_hat2 = self.d_net2(self.x_hat2)
        self.grad_x_hat2 = tf.gradients(self.D_x_hat2, self.x_hat2)[0]
        self.grad_x_hat_l2_2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat2), axis=1))
        self.gradient_penalty2 =  tf.square(self.grad_x_hat_l2_2 - 1.0)

        self. w_distance2 = tf.reduce_mean(self.D_G_zi2) - tf.reduce_mean(self.D_xi2) 
        grad_penal2 =  args.lambda_g2 *tf.reduce_mean(self.gradient_penalty2 )
        self.D_loss2 = self. w_distance2 + grad_penal2

        self.l1_loss1 =tf.reduce_mean(tf.abs(self.G_zi1 - self.x_i1))  
        self.l1_loss2 =tf.reduce_mean(tf.abs(self.G_zi2 - self.x_i2))  

        self.G_loss1 = - args.lambda_w1 * tf.reduce_mean(self.D_G_zi1)+ args.lambda_1 * self.l1_loss1+ args.lambda_gd1*self.gdl_loss1
        self.G_loss2 = - args.lambda_w2 * tf.reduce_mean(self.D_G_zi2)+ args.lambda_2 * self.l1_loss2+ args.lambda_gd2*self.gdl_loss2

        t_vars = tf.trainable_variables()
		self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars1 = [var for var in t_vars if 'discriminator1' in var.name]
        self.d_vars2 = [var for var in t_vars if 'discriminator2' in var.name]
        
        self.summary_d_loss_all1 = tf.summary.scalar("1_DiscriminatorLoss", self.D_loss1)
        self.summary_d_loss_all2 = tf.summary.scalar("2_DiscriminatorLoss", self.D_loss2)
        self.summary_d_loss_1_1 = tf.summary.scalar("1_w_distance",  self.w_distance1)
        self.summary_d_loss_1_2 = tf.summary.scalar("2_w_distance",  self.w_distance2)

        self.summary_l1_loss_1 = tf.summary.scalar("1_l1_loss", self.l1_loss1)
        self.summary_l1_loss_2 = tf.summary.scalar("2_l1_loss", self.l1_loss2)
        self.summary_g_loss1 = tf.summary.scalar("GeneratorLoss1", self.G_loss1)
        self.summary_g_loss2 = tf.summary.scalar("GeneratorLoss2", self.G_loss2)
        self.summary_gdl_loss_1 = tf.summary.scalar("1_gdl_loss",self.gdl_loss1)
        self.summary_gdl_loss_2 = tf.summary.scalar("2_gdl_loss",self.gdl_loss2)
        
        self.summary_all_loss1 = tf.summary.merge([ self.summary_d_loss_all1, self.summary_d_loss_1_1,\
                                                  self.summary_l1_loss_1, self.summary_g_loss1,self.summary_gdl_loss_1])
        self.summary_all_loss2 = tf.summary.merge([ self.summary_d_loss_all2, self.summary_d_loss_1_2,\
                                                  self.summary_l1_loss_2, self.summary_g_loss2,self.summary_gdl_loss_2])

        self.d_adam1, self.g_adam1, self.d_adam2, self.g_adam2 = None, None,None,None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam1 = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.D_loss1, var_list = self.d_vars1)
            self.g_adam1 = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.G_loss1, var_list = self.g_vars)
            self.d_adam2 = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.D_loss2, var_list = self.d_vars2)
            self.g_adam2 = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.G_loss2, var_list = self.g_vars)
        self.saver = tf.train.Saver(max_to_keep=None)
        print('--------------------------------------------\n# of parameters : {} '.\
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    def train(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.start_step = 0 
        if args.continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")      
        print('Start point : iter : {}'.format(self.start_step))
        start_time = time.time()
        image_batch_v1, label_batch_v1, image_batch_v2, label_batch_v2,threads=self.image_loader.input_pipeline(self.sess,args.batch_size,args.whole_size,args.patch_size)
   		
		for _i in range(0, args.d_iters):
                 
                    tra_images1,tra_labels1=self.sess.run([image_batch_v1, label_batch_v1])
                    tra_images2,tra_labels2=self.sess.run([image_batch_v2, label_batch_v2])   
					        
                    self.sess.run(self.d_adam1,feed_dict={self.z_i1: tra_images1,self.x_i1:tra_labels1})
                    self.sess.run(self.d_adam2,feed_dict={self.z_i2: tra_images2,self.x_i2:tra_labels2})
					
		tra_images1,tra_labels1=self.sess.run([image_batch_v1, label_batch_v1])
        tra_images2,tra_labels2=self.sess.run([image_batch_v2, label_batch_v2])
           
        a1, summary_str1= self.sess.run([self.g_adam1,  self.summary_all_loss1],\
            feed_dict={self.z_i1: tra_images1,self.x_i1:tra_labels1,self.z_i2: tra_images2,self.x_i2:tra_labels2})
        a2, summary_str2= self.sess.run([self.g_adam2, self.summary_all_loss2],\
            feed_dict={self.z_i2: tra_images2,self.x_i2:tra_labels2,self.z_i2: tra_images2,self.x_i2:tra_labels2})		
		g_loss1, g_loss2 =self.sess.run([ self.G_loss1,self.G_loss2],\
            feed_dict={self.z_i1: tra_images1,self.x_i1:tra_labels1,self.z_i2: tra_images2,self.x_i2:tra_labels2})
			
        for  iteration in range(args.num_epoch):
           
            for t in range(0, args.num_iter):
           
                for _ in range(0, args.d_iters):
                 
                    tra_images1,tra_labels1=self.sess.run([image_batch_v1, label_batch_v1])
                    tra_images2,tra_labels2=self.sess.run([image_batch_v2, label_batch_v2])   
					        
                    self.sess.run(self.d_adam1,feed_dict={self.z_i1: tra_images1,self.x_i1:tra_labels1})
                    self.sess.run(self.d_adam2,feed_dict={self.z_i2: tra_images2,self.x_i2:tra_labels2})
            
 				if ((g_loss1 - g_loss2) >= (args.lc1 - args.lc2) ):
				
					tra_images1,tra_labels1=self.sess.run([image_batch_v1, label_batch_v1])
                		a1, summary_str1= self.sess.run([self.g_adam1,  self.summary_all_loss1],\
                    feed_dict={self.z_i1: tra_images1,self.x_i1:tra_labels1,self.z_i2: tra_images2,self.x_i2:tra_labels2})
					self.writer.add_summary(summary_str1, (iteration+1)*t)
					
				else:
                		tra_images2,tra_labels2=self.sess.run([image_batch_v2, label_batch_v2])
            			a2, summary_str2= self.sess.run([self.g_adam2, self.summary_all_loss2],\
                    feed_dict={self.z_i2: tra_images2,self.x_i2:tra_labels2,self.z_i2: tra_images2,self.x_i2:tra_labels2})
   					self.writer.add_summary(summary_str2, (iteration+1)*t)

                if (t+1) % args.print_freq == 0:
                    d_loss1, g_loss1,  W_distance1 ,g_zi_img1 ,d_loss2, g_loss2,  W_distance2,g_zi_img2 = \
                    self.sess.run([self.D_loss1, self.G_loss1, self. w_distance1,self.G_zi1,self.D_loss2, self.G_loss2, self. w_distance2,self.G_zi2],\
                    feed_dict={self.z_i1: tra_images1,self.x_i1:tra_labels1,self.z_i2: tra_images2,self.x_i2:tra_labels2})
                    print('Epoch:{} Iter {} Time {} d_loss1 {} g_loss1 {} W_distance1{} d_loss2 {} g_loss2 {} W_distance2{}'.format(iteration,t, time.time() - start_time,\
                     d_loss1, g_loss1,W_distance1,d_loss2, g_loss2,W_distance2))
            if (iteration+1) % args.save_freq == 0:
                self.save(args, iteration)  
            print('Epoch')
        self.image_loader.coord.request_stop()
        self.image_loader.coord.join(threads)
        self.sess.close()

    def save(self, args, step):
        model_name = args.model + ".ckpt"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,\
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.start_step = int(ckpt_name.split('-')[-1])
            meta_name=os.path.join(self.checkpoint_dir,ckpt_name+".meta")
            self.saver = tf.train.import_meta_graph(meta_name)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False
    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
  
        npy_save_dir = os.path.join('.', args.test_npy_save_dir, self.p_info)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)
        test_append1=[]
        test_append2=[]
        a=len(self.test_image_loader.input_images1)
        for idx in range(a):
            
            test_zi1, test_xi1 = self.test_image_loader.input_images1[idx], self.test_image_loader.label_images1[idx]
            test_zi2, test_xi2 = self.test_image_loader.input_images2[idx], self.test_image_loader.label_images2[idx]
            
            whole_G_zi1 = self.sess.run(self.G_whole_zi1, feed_dict={self.whole_z1: test_zi1.reshape(self.whole_z1.get_shape().as_list())})
            whole_G_zi2 = self.sess.run(self.G_whole_zi2, feed_dict={self.whole_z2: test_zi2.reshape(self.whole_z2.get_shape().as_list())})

            yy1=whole_G_zi1[0,:,:,:]
            plt.figure(figsize=(10,5))
            plt.imshow(yy1[:,:,0],cmap='gray'), plt.axis('off')
            yyt1=np.transpose(yy1,(2,1,0))
            test_append1.append(yyt1)
            array1=np.array(test_append1)


            yy2=whole_G_zi2[0,:,:,:]
            plt.figure(figsize=(10,5))
            plt.imshow(yy2[:,:,0],cmap='gray'), plt.axis('off')
            yyt2=np.transpose(yy2,(2,1,0))
            test_append2.append(yyt2)
            array2=np.array(test_append2)
            
        yytf1=np.reshape(array1,[1,args.whole_size*args.whole_size*1*a])
        yytf2=np.reshape(array2,[1,args.whole_size*args.whole_size*1*a])
            
        fid1=open('test1.bin','wb')
        fid2=open('test2.bin','wb')
        fid1.write(yytf1)
        fid2.write(yytf2)
        fid1.close()
        fid2.close()