#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:30:35 2021

@author: magictjc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat 
import math
from ops import *



class estimate():
    
    def __init__(self,data,data_tile,latent_dim_x,latent_dim_h,cluster_k,ECx_DIM,ECh_DIM,c_dim,learning_rate,
                 train_steps,standard_signal,sample_point,trans_signal,pilot_data,detect_phase_data,MLstan,detect_trans_signal,perfect_esti,MMSE_diagterm):
        
        self.data = data.astype('float32')
        self.data_tile = data_tile.astype('float32')
        self.k = cluster_k
        self.x_dim = latent_dim_x     #8
        self.h_dim = latent_dim_h     #16
        self.ecx_dim = ECx_DIM
        self.ech_dim = ECh_DIM
        self.c_dim = c_dim
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.standard_signal = standard_signal.astype('float32')
        self.sample_point = sample_point
        self.trans_signal = trans_signal.astype('float32')
        self.pilot_data = pilot_data
        self.detect_phase_data = detect_phase_data.astype('complex64')
        self.MLstan = MLstan.astype('complex64')
        self.detect_trans_signal = detect_trans_signal.astype('complex64')
        self.perfect_esti =perfect_esti.astype('complex64')
        self.MMSE_diagterm = MMSE_diagterm

    def encoder_x(self):
        
        h = self.input_y                                # q_theta_under_y
        
        for idx,dim in enumerate(self.ecx_dim):          # idx 为 self.ec_dim 的数据下标; dim 为self.ec_dim的数据
            layer_name = 'ecx_' + str(idx)
            h = tf.nn.tanh(fully_connected(h,dim,name=layer_name))
            
        x_mean = fully_connected(h,self.x_dim,name='x_mean')
        x_log_var = fully_connected(h,self.x_dim,name='x_log_var')
        diag_x_log_var=tf.matrix_diag(tf.exp(x_log_var))
        
        
        x_mean_rreshape = tf.expand_dims(x_mean,2)                               #100*8-->100*8*1
        slice1,slice2 = tf.split(x_mean_rreshape,2,1)                            #100*8*1-->100*4*1  100*4*1
        x_mean_sliceconcat = tf.concat([slice1,slice2],2)                        #100*4*1 100*4*1 --> 100*4*2
        x_mean_square = tf.reduce_sum(tf.square(x_mean_sliceconcat),2)           #100*4*2 --> 100*4
        x_mean_square_sqrt = tf.sqrt(x_mean_square)
        x_mean_square_sqrt_tile = tf.tile(tf.expand_dims(x_mean_square_sqrt,2),[1,1,2]) # 100*4 --> 100*4*2
        x_mean_divide = tf.divide(x_mean_sliceconcat,x_mean_square_sqrt_tile)
        slice11,slice22 = tf.split(x_mean_divide,2,2)    # 100*4*2 -->  100*4*1  100*4*1
        x_mean_test = tf.concat([slice11,slice22],1)
        x_mean = tf.reduce_mean(x_mean_test,2)
        
        #sslice1,sslice2,sslice3,sslice4 = tf.split(x_mean,4,1)      #x分成4列
        #slice_contact1 = tf.concat([sslice1,sslice3],1)     
        #slice_contact2 = tf.concat([sslice2,sslice4],1)
        #slice_contact = tf.concat([slice_contact1,slice_contact2],0)
        slice_contact=0
       
        return x_mean,x_log_var,diag_x_log_var,slice_contact
    
    def encoder_h(self):
        
        h = self.input_y_tile                                # q_theta_under_y
        
        for idx,dim in enumerate(self.ech_dim):          # idx 为 self.ec_dim 的数据下标; dim 为self.ec_dim的数据
            layer_name = 'ech_' + str(idx)
            h = tf.nn.tanh(fully_connected(h,dim,name=layer_name))
            
        h_real_mean = fully_connected(h,self.h_dim,name='z_real_mean')
        h_real_log_var = fully_connected(h,self.h_dim,name='z_real_log_var')
        
        h_imag_mean = fully_connected(h,self.h_dim,name='z_imag_mean')
        h_imag_log_var = fully_connected(h,self.h_dim,name='z_imag_log_var')
        
       
        return h_real_mean,h_real_log_var,h_imag_mean,h_imag_log_var

    def sample(self,args):                             
        
        h_real_mean,h_real_log_var = args
        h_real_mean_tail = tf.tile(tf.expand_dims(h_real_mean,axis=0),[self.sample_point,1,1])
        h_real_log_var_tail = tf.tile(tf.expand_dims(h_real_log_var,axis=0),[self.sample_point,1,1])
        
        eps = tf.random_normal(shape=[self.sample_point,tf.shape(self.data)[0],self.h_dim])    #random_normal(shape=[samplepoint,datapoint,16])
        h_samples = tf.exp(h_real_log_var_tail/2) * eps + h_real_mean_tail                     #(shape=[samplepoint,datapoint,16])
        
        
        return h_samples
    

    def model(self):
        
        tf.reset_default_graph()
        
        self.input_y = tf.placeholder(dtype=tf.float32,
                            shape=[None,self.data.shape[1]],name='input_y')     #shape = [none,2]
        self.input_y_tile = tf.placeholder(dtype=tf.float32,
                            shape=[None,self.data_tile.shape[1]],name='input_y_tile')     #shape = [none,2]

        
        self.x_mean,self.x_log_var,self.diag_x_log_var,self.figure_data = self.encoder_x()
        

        self.h_real_mean,self.h_real_log_var,self.h_imag_mean,self.h_imag_log_var = self.encoder_h()

        self.y = tf.tile(tf.expand_dims(self.data,axis=1),[1,self.standard_signal.shape[0],1])           # 1000*16*2  这个地方应该再考虑数据维度放置
        


        h_real_sample = (self.sample([self.h_real_mean,self.h_real_log_var]))
        
        h_imag_sample = (self.sample([self.h_imag_mean,self.h_imag_log_var]))
        h_real_reshape = tf.reshape(h_real_sample,[self.sample_point,self.data.shape[0],data.shape[1],1])     #100*100*4-->100*100*4*1
        h_imag_reshape = tf.reshape(h_imag_sample,[self.sample_point,self.data.shape[0],data.shape[1],1])
        h_concat1=tf.concat([h_real_reshape, h_imag_reshape], 2)                #100*100*4*4-->100*100*8*4
        h_concat2=tf.concat([-h_imag_reshape, h_real_reshape], 2)               #100*100*4*4-->100*100*8*4
        h_concatres=tf.concat([h_concat1, h_concat2], 3)                        #100*100*8*4-->100*100*8*8 channel matrix
        h_concatres_transpose = tf.transpose(h_concatres,[0,1,3,2])   
        
        x_mean_reshape = tf.reshape(self.x_mean,[self.data.shape[0],trans_signal.shape[1],1])       #100*8*1 
        x_mean_reshape_transpose = tf.transpose(x_mean_reshape,[0,2,1])         #100*1*8
        x_mean_reshape_tail = tf.tile(tf.expand_dims(x_mean_reshape,axis=0),[self.sample_point,1,1,1])                     #100*100*8*1 
        x_mean_reshape_reanspose_tail = tf.tile(tf.expand_dims(x_mean_reshape_transpose,axis=0),[self.sample_point,1,1,1]) #100*100*1*8 
        diag_x_log_var_tail = tf.tile(tf.expand_dims(self.diag_x_log_var,axis=0),[self.sample_point,1,1,1])
        
        self.h_concatres =tf.reduce_mean(tf.reduce_mean(h_concatres,0),0)
        

        
        h_mean_concat = tf.concat([self.h_real_log_var,self.h_imag_log_var],1)                        #100*32
        h_mean_concat_reshape = tf.reshape(h_mean_concat,[self.data.shape[0],h_mean_concat.shape[1],1]) #100*32*1
        h_mean_concat_reshape_transpose = tf.transpose(h_mean_concat_reshape,[0,2,1])                   #100*1*32
        
        h_var_concat = tf.concat([self.h_real_log_var,self.h_imag_log_var],1)                           #100*32
        
        
        distance = tf.reduce_mean(tf.square(tf.reduce_mean(tf.matmul(h_concatres,x_mean_reshape_tail),3) - tf.tile(tf.expand_dims(self.data,axis=0),[self.sample_point,1,1])),0)
        distance =tf.reduce_sum(distance,1)
        
        trace=tf.reduce_mean(tf.trace(tf.matmul(h_concatres,tf.matmul(diag_x_log_var_tail,h_concatres_transpose))),0)
        
                
        x_mean_rrrreshape = tf.expand_dims(self.x_mean,2)                               #100*8-->100*8*1
        slice1111,slice2222 = tf.split(x_mean_rrrreshape,2,1)                            #100*8*1-->100*4*1  100*4*1
        average =  tf.square(slice1111)-tf.square(slice2222)
        self.average = tf.reduce_mean(tf.square(average))*0.1
        self.average = 0
        
        


        self.loss_1 =-0.5*tf.reduce_mean((tf.reduce_sum(self.x_log_var,1)+tf.reduce_sum(self.h_real_log_var,1)+tf.reduce_sum(self.h_imag_log_var,1)))
        
        self.loss_2 =0.5*(tf.reduce_sum(tf.exp(h_var_concat)) + tf.reduce_sum(tf.matmul(h_mean_concat_reshape_transpose,h_mean_concat_reshape))+ \
                      tf.reduce_sum(tf.exp(self.x_log_var))+ tf.reduce_sum(tf.matmul(x_mean_reshape_transpose,x_mean_reshape)))/(self.data.shape[0])
        
        self.loss_3 =tf.reduce_sum(trace+distance)/(self.data.shape[0])
        

    
    
        self.h_var = tf.reduce_mean(tf.square(h_real_sample-tf.reduce_mean(h_real_sample)))
     
        vade_loss =self.loss_1 + self.loss_2 + self.loss_3+ self.average 
        
        self.lower_bound = self.loss_1 + self.loss_2 + self.loss_3 + self.average 
        
        self.xtest = abs(self.x_mean)
        
        
        xtest_slice1,xtest_slice2 = tf.split(self.xtest,2,1)                        #将xtest分成两列
        xtest_slice1_tail = tf.tile(tf.expand_dims(xtest_slice1,axis=1),[1,2,1])    #对两列分别进行判决
        xtest_slice2_tail = tf.tile(tf.expand_dims(xtest_slice2,axis=1),[1,2,1])
    
        sound_trans_signal_tail = tf.tile(tf.expand_dims(tf.transpose(self.trans_signal),axis=0),[self.data.shape[0],1,1])
        
        metric1_test = tf.reduce_sum(xtest_slice1_tail - sound_trans_signal_tail,2)
        metric2_test = tf.reduce_sum(xtest_slice2_tail - sound_trans_signal_tail,2)
        decision1 =  tf.cast(tf.argmin(abs(metric1_test),1),dtype=tf.float32)
        decision2 =  tf.cast(tf.argmin(abs(metric2_test),1),dtype=tf.float32)
        
        data_aided = tf.complex(tf.constant([0.0]),tf.constant([1.0]))
        matrix_data_aided = tf.matrix_diag(tf.concat([data_aided,data_aided,data_aided,data_aided],0)) #正交导频

        
       
        
        self.decision_contact = tf.concat([tf.reshape(decision1,[decision1.shape[0],1]),tf.reshape(decision2,[decision2.shape[0],1])],1) #拼接判决结果
        reorder = tf.reduce_sum(self.decision_contact,0)   #调整保证顺序一致
        self.decision_contact = tf.cond(reorder[0]>reorder[1],lambda:tf.concat([tf.reshape(decision2,[decision2.shape[0],1]),tf.reshape(decision1,[decision1.shape[0],1])],1),lambda:self.decision_contact) #拼接判决结果)
        
        real_decision_contact,imag_decision_contact=tf.split(self.decision_contact,2,1)
        self.complex_decision = tf.complex(real_decision_contact,imag_decision_contact)
        
        decision_slice1,decision_slice2,decision_slice3,decision_slice4 = tf.split(self.complex_decision,4,0)
        decision_slice1 = tf.reduce_mean(decision_slice1,0)
        decision_slice2 = tf.reduce_mean(decision_slice2,0)
        decision_slice3 = tf.reduce_mean(decision_slice3,0)
        decision_slice4 = tf.reduce_mean(decision_slice4,0)
        
        pilot_data1,pilot_data2,pilot_data3,pilot_data4 = tf.split(self.pilot_data,4,0)
        pilot_data1 = tf.reshape(pilot_data1[0,:],[self.pilot_data.shape[1],1])
        pilot_data2 = tf.reshape(pilot_data2[0,:],[self.pilot_data.shape[1],1])
        pilot_data3 = tf.reshape(pilot_data3[0,:],[self.pilot_data.shape[1],1])
        pilot_data4 = tf.reshape(pilot_data4[0,:],[self.pilot_data.shape[1],1])
        self.pilot_data = tf.concat([pilot_data1,pilot_data2,pilot_data3,pilot_data4],1)

        
        '''
        blind esti
        '''
        matrix_x=tf.matrix_diag(tf.concat([decision_slice1,decision_slice2,decision_slice3,decision_slice4],0))
        #self.test = tf.transpose(tf.conj(matrix_x))
        XHXX = tf.matmul(tf.transpose(tf.conj(matrix_x)),tf.matrix_inverse(tf.matmul(matrix_x,tf.transpose(tf.conj(matrix_x)))+self.MMSE_diagterm))
        #self.test = tf.matmul(tf.cast(self.pilot_data,dtype=tf.complex64),XHXX)
        self.matrix_x = matrix_x
        self.yyy = tf.cast(self.pilot_data,dtype=tf.complex64)
        self.XHXX = XHXX
        self.blind_esti = tf.transpose(tf.matmul(self.yyy,XHXX))
        self.blind_CSI_esti_error = tf.square((abs(abs(self.blind_esti)  - abs(self.perfect_esti))))#/abs(self.perfect_esti)

        
        
        HYHH = tf.matmul(tf.transpose(tf.conj(self.blind_esti)),tf.matrix_inverse(tf.matmul(self.blind_esti,tf.transpose(tf.conj(self.blind_esti)))))
        self.blind_zf = tf.transpose(tf.matmul(tf.cast(self.detect_phase_data,dtype=tf.complex64), HYHH))
        
        
        '''
        data_aided esti LS/ML
        '''
        XHXX_data_aided = tf.matmul(tf.transpose(tf.conj(matrix_data_aided)),tf.matrix_inverse(tf.matmul(matrix_data_aided,tf.transpose(tf.conj(matrix_data_aided)))))
        self.data_aided_esti = tf.transpose(tf.matmul(self.yyy,XHXX_data_aided))
        self.aided_CSI_esti_error = tf.square((abs(abs(self.data_aided_esti)  - abs(self.perfect_esti))))#/abs(self.perfect_esti)
        
        '''
        data_aided esti MMSE
        '''
        XHXX_data_aided_MMSE = tf.matmul(tf.transpose(tf.conj(matrix_data_aided)),tf.matrix_inverse(tf.matmul(matrix_data_aided,tf.transpose(tf.conj(matrix_data_aided))) + self.MMSE_diagterm))
        self.data_aided_MMSE_esti = tf.transpose(tf.matmul(self.yyy,XHXX_data_aided_MMSE))
        self.aided_CSI_MMSE_esti_error = tf.square((abs(abs(self.data_aided_MMSE_esti)  - abs(self.perfect_esti))))#/abs(self.perfect_esti)
        
        MLstan_time_aided_H = tf.matmul(self.MLstan,self.data_aided_esti)
        MLstan_time_aided_MMSE_H = tf.matmul(self.MLstan,self.data_aided_MMSE_esti)
        self.MLstan_time_aided_H=abs(MLstan_time_aided_H)
        MLstan_time_aided_H_tail = tf.tile(tf.expand_dims(MLstan_time_aided_H,axis=0),[self.detect_phase_data.shape[0],1,1])
        MLstan_time_aided_MMSE_H_tail = tf.tile(tf.expand_dims(MLstan_time_aided_MMSE_H,axis=0),[self.detect_phase_data.shape[0],1,1])
        MLstan_time_blind_H = tf.matmul(self.MLstan,self.blind_esti)
        MLstan_time_blind_H_tail = tf.tile(tf.expand_dims(MLstan_time_blind_H,axis=0),[self.detect_phase_data.shape[0],1,1])
        MLstan_time_perfect_H  =  tf.matmul(self.MLstan,self.perfect_esti)
        MLstan_time_perfect_H_tail = tf.tile(tf.expand_dims(MLstan_time_perfect_H,axis=0),[self.detect_phase_data.shape[0],1,1])
        
        detect_phase_data_tail =tf.tile(tf.expand_dims(self.detect_phase_data,axis=1),[1,self.MLstan.shape[0],1]) 
        
        aided_distance = tf.reduce_sum(abs(MLstan_time_aided_H_tail - detect_phase_data_tail),2)
        self.aided_min_index = tf.argmin(aided_distance,1)
        
        
        aided_MMSE_distance = tf.reduce_sum(abs(MLstan_time_aided_MMSE_H_tail - detect_phase_data_tail),2)
        self.aided_MMSE_min_index = tf.argmin(aided_MMSE_distance,1)
        
        
        blind_distance = tf.reduce_sum(abs(MLstan_time_blind_H_tail - detect_phase_data_tail),2)
        self.blind_min_index = tf.argmin(blind_distance,1)
        
        perfect_distance = tf.reduce_sum(abs(MLstan_time_perfect_H_tail - detect_phase_data_tail),2)
        self.perfect_min_index = tf.argmin(perfect_distance,1)

        


        
        return vade_loss
    
    
    
    def train(self):
        
        
        vade_loss = self.model()
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(vade_loss)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())    #变量初始化
        
                

        for step in range(self.train_steps):
            
            
            
            self.sess.run(train_op,feed_dict={self.input_y:self.data,self.input_y_tile:self.data_tile})
            if step % 10 == 0:
                
                current_loss = self.lower_bound.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                loss_1 = self.loss_1.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                loss_2 = self.loss_2.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                loss_3 = self.loss_3.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #xtest = self.xtest.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                decision_contact = self.decision_contact.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #blind_esti = self.blind_esti.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #data_aided_esti = self.data_aided_esti.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #yyy = self.yyy.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #XHXX = self.XHXX.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #matrix_x = self.matrix_x.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                
                
                
                blind_CSI_esti_error = self.blind_CSI_esti_error.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                Mean_blind_CSI_esti_error = np.mean(blind_CSI_esti_error)
                Var_blind_CSI_esti_error = np.var(blind_CSI_esti_error)
                print("Mean_blind_CSI_esti_error is {Mean_blind_CSI_esti_error}".format(Mean_blind_CSI_esti_error=Mean_blind_CSI_esti_error))
                #print("Var_blind_CSI_esti_error is {Var_blind_CSI_esti_error}".format(Var_blind_CSI_esti_error=Var_blind_CSI_esti_error))
                
                aided_CSI_esti_error =self.aided_CSI_esti_error.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                Mean_aided_CSI_esti_error = np.mean(aided_CSI_esti_error)
                Var_aided_CSI_esti_error = np.var(aided_CSI_esti_error)
                print("Mean_aided_CSI_esti_error is {Mean_aided_CSI_esti_error}".format(Mean_aided_CSI_esti_error=Mean_aided_CSI_esti_error))
                #print("Var_aided_CSI_esti_error is {Var_aided_CSI_esti_error}".format(Var_aided_CSI_esti_error=Var_aided_CSI_esti_error))
                
                aided_CSI_MMSE_esti_error =self.aided_CSI_MMSE_esti_error.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                Mean_aided_CSI_MMSE_esti_error = np.mean(aided_CSI_MMSE_esti_error)
                Var_aided_CSI_MMSE_esti_error = np.var(aided_CSI_MMSE_esti_error)
                print("Mean_aided_CSI_MMSE_esti_error is {Mean_aided_CSI_MMSE_esti_error}".format(Mean_aided_CSI_MMSE_esti_error=Mean_aided_CSI_MMSE_esti_error))
                #print("Var_aided_CSI_MMSE_esti_error is {Var_aided_CSI_MMSE_esti_error}".format(Var_aided_CSI_MMSE_esti_error=Var_aided_CSI_MMSE_esti_error))
                
                
                blind_zf = self.blind_zf.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                self.blind_zfzf = blind_zf
                
                
                
                aided_min_index = self.aided_min_index.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                aided_dete_result = np.expand_dims(self.MLstan[aided_min_index[1]],1)
                for index in range(aided_min_index.shape[0]):
                    aided_dete_result = np.append(aided_dete_result,np.expand_dims(self.MLstan[aided_min_index[index]],1),1)
                    
                aided_dete_result = np.transpose(aided_dete_result)
                aided_dete_result = aided_dete_result[1:,:]
                aided_difference = len(np.argwhere(aided_dete_result - self.detect_trans_signal))/(self.detect_trans_signal.shape[0]*4)
                print("aided_difference is {aided_difference}".format(aided_difference=aided_difference))
                
                
                blind_min_index = self.blind_min_index.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                blind_dete_result = np.expand_dims(self.MLstan[blind_min_index[1]],1)
                for index in range(blind_min_index.shape[0]):
                    blind_dete_result = np.append(blind_dete_result,np.expand_dims(self.MLstan[blind_min_index[index]],1),1)
                    
                blind_dete_result = np.transpose(blind_dete_result)
                blind_dete_result = blind_dete_result[1:,:]
                blind_difference = len(np.argwhere(blind_dete_result - self.detect_trans_signal))/(self.detect_trans_signal.shape[0]*4)
                self.blind_dete_result = blind_dete_result
                print("blind_difference is {blind_difference}".format(blind_difference=blind_difference))
                
                
                perfect_min_index = self.perfect_min_index.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                perfect_dete_result = np.expand_dims(self.MLstan[perfect_min_index[1]],1)
                for index in range(perfect_min_index.shape[0]):
                    perfect_dete_result = np.append(perfect_dete_result,np.expand_dims(self.MLstan[perfect_min_index[index]],1),1)
                    
                perfect_dete_result = np.transpose(perfect_dete_result)
                perfect_dete_result = perfect_dete_result[1:,:]
                perfect_difference = len(np.argwhere(perfect_dete_result - self.detect_trans_signal))/(self.detect_trans_signal.shape[0]*4)
                self.perfect_dete_result = perfect_dete_result
                print("perfect_difference is {perfect_difference}".format(perfect_difference=perfect_difference))
                
                
                aided_MMSE_min_index = self.aided_MMSE_min_index.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                aided_MMSE_dete_result = np.expand_dims(self.MLstan[aided_MMSE_min_index[1]],1)
                for index in range(aided_MMSE_min_index.shape[0]):
                    aided_MMSE_dete_result = np.append(aided_MMSE_dete_result,np.expand_dims(self.MLstan[aided_MMSE_min_index[index]],1),1)
                    
                aided_MMSE_dete_result = np.transpose(aided_MMSE_dete_result)
                aided_MMSE_dete_result = aided_MMSE_dete_result[1:,:]
                aided_MMSE_difference = len(np.argwhere(aided_MMSE_dete_result - self.detect_trans_signal))/(self.detect_trans_signal.shape[0]*4)
                self.aided_MMSE_dete_result = aided_MMSE_dete_result
                print("aided_MMSE_difference is {aided_MMSE_difference}".format(aided_MMSE_difference=aided_MMSE_difference))
                
                
             
                

                
                #h_concatresh_concatres = self.h_concatres.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
                #self.figure_dataeval= self.figure_data.eval({self.input_y:self.data,self.input_y_tile:self.data_tile})
               

                #print(self.pi.eval({self.input_y:self.data}))
                print("After {step} training, the current EBLO is {current_loss}, \
                      the loss_1 is {loss_1}, the loss_2 is {loss_2}, \
                      the loss_3 is {loss_3}"
                      .format(step=step,current_loss=current_loss,loss_1=loss_1,
                        loss_2=loss_2,loss_3=loss_3))
                #print("xtest is {xtest}".format(xtest=xtest))
                print("decision_contact is {decision_contact}".format(decision_contact=decision_contact))
                #print("blind_esti is {blind_esti}".format(blind_esti=blind_esti))
                #print("data_aided_esti is {data_aided_esti}".format(data_aided_esti=data_aided_esti))
                #print("aided_min_index is {aided_min_index}".format(aided_min_index=aided_min_index))
                
                
                #print("matrix_x is {matrix_x}".format(matrix_x=matrix_x))
                #print("yyy is {yyy}".format(yyy=yyy))
                #print("XHXX is {XHXX}".format(XHXX=XHXX))
                #print("h_real_mean is {h_real_mean}".format(h_real_mean=h_real_mean))
                #print("h_concatresh_concatres is {h_concatresh_concatres}".format(h_concatresh_concatres=h_concatresh_concatres))
                idx = int(step/10)
                
                self.figure(idx) 
                
                                                                                                                                           #self.figure(idx)
                
        
        
    def figure(self,idx):
        
        tf.get_variable_scope().reuse_variables()
        
        
        plt.figure(idx, figsize=(3,3),dpi=30)
        plt.grid()
#       plt.scatter(curve[:,0],curve[:,1],c='b',marker='.',linewidths=0)
        plt.xlabel('I')
        plt.ylabel('Q')
        
        before_equl = np.reshape(self.detect_phase_data,[self.detect_phase_data.shape[0]*self.detect_phase_data.shape[1]]) 
        
        
        
        plt.scatter(np.real(before_equl),np.imag(before_equl),marker='.',linewidths=0)
        #plt.scatter(self.curve[:,:,0],self.curve[:,:,1],c='g', s=5)
        plt.show()
        
        
        plt.figure(figsize=(3,3),dpi=30)
        plt.grid()

        plt.xlabel('I')
        plt.ylabel('Q')
        
        blind_dete_result =  np.reshape(self.blind_zfzf,[self.blind_zfzf.shape[0]*self.blind_zfzf.shape[1]]) 
        plt.scatter(np.real(blind_dete_result),np.imag(blind_dete_result),marker='.',linewidths=0)

        plt.show()
        







dict = loadmat("/Users/magictjc/OneDrive/博士期间/mimo/received_signal.mat")
data = dict['received_signal']
trans_signal = dict['trans_signal']
standard_signal = dict['standard_signal']
MLstan = dict['MLstan']
detect_trans_signal  = dict['detect_trans_signal']
perfect_esti =  dict['h_test']
MMSE_diagterm = dict['MMSE_diagterm']
trans_signal = np.hstack((trans_signal.real,trans_signal.imag))
sounding_phase_data = data[0:20,:]      #'''提取sounding phase 信号'''
detect_phase_data = data[20:,:]     # '''提取detecting phase 信号'''
train_data = np.hstack((sounding_phase_data.real,sounding_phase_data.imag))
pilot_data =  data[0:20,:]

standard_signal = np.hstack((standard_signal.real,standard_signal.imag))
train_data_reshape = np.reshape(train_data,[1,train_data.shape[0]*train_data.shape[1]])
train_data_reshape_tile = np.tile(train_data_reshape,[train_data.shape[0],1])

#ECx_DIM = [8,8]
#ECh_DIM = [8,8]
ECx_DIM = [16]
ECh_DIM = [16]
C_DIM = [8,8]

#train_vade = estimate(train_data,train_data_reshape_tile,train_data.shape[1],data.shape[1]*data.shape[1],4,ECx_DIM,ECh_DIM,C_DIM,0.5e-1,350000,standard_signal,20)
train_vade = estimate(train_data,train_data_reshape_tile,trans_signal.shape[1],data.shape[1],4,ECx_DIM,ECh_DIM,C_DIM,0.5e-2,200,standard_signal,10,trans_signal,pilot_data,detect_phase_data,MLstan,detect_trans_signal,perfect_esti,MMSE_diagterm)

train_vade.train()
