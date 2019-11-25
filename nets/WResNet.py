from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import Multiple

from nets import Distillation as Dist

tcf = tf.contrib.framework
tcl = tf.contrib.layers

def BasicBlock(x, out_planes, stride = 2, name = None):
    with tf.variable_scope(name):
        equalInOut = x.get_shape().as_list()[-1] == out_planes
        if not equalInOut:
            x_ = tcl.batch_norm(x, scope='bn')
            x = tf.nn.relu(x_)
        else:
            out_ = tcl.batch_norm(x, scope='bn')
            out = tf.nn.relu(out_)
                
        out = tcl.batch_norm(tcl.conv2d(out if equalInOut else x, out_planes, [3,3], stride, scope='conv0'), activation_fn = tf.nn.relu, scope='bn0')
#        out = tcl.dropout(out, 0.7)

        out = tcl.conv2d(out, out_planes, [3,3], 1, scope='conv1')
        if not(equalInOut):
            x = tcl.conv2d(x, out_planes, [1,1], stride, scope='conv2')
        return x+out
    
def NetworkBlock(x, nb_layers, out_planes, stride, name = ''):
    with tf.variable_scope(name):
        for i in range(nb_layers):           
            x = BasicBlock(x, out_planes, stride = stride if i == 0 else 1, name = 'BasicBlock%d'%i)
        return x

def WResNet(image, label, scope, is_training, Distill = None):
    end_points = {}

    if scope == 'Teacher':     
        depth = 40; widen_factor = 2
    else:
        depth = 16; widen_factor = 1
    
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    stride = [1,2,2]
    n = (depth-4)//6
    
    with tf.variable_scope(scope):
        with tcf.arg_scope([tcl.conv2d, tcl.fully_connected, tcl.batch_norm], trainable = True):
            with tcf.arg_scope([tcl.dropout, tcl.batch_norm], is_training = is_training):
                std = tcl.conv2d(image, nChannels[0], [3,3], 1, scope='conv0')
                tf.add_to_collection('feat', std)
                for i in range(3): 
                    std = NetworkBlock(std, n, nChannels[1+i], stride[i], name = 'WResblock%d'%i)
                    tf.add_to_collection('feat', std)
                std = tcl.batch_norm(std, scope='bn0')
                std = tf.nn.relu(std)

                fc = tf.reduce_mean(std, [1,2])
                
                logits = tcl.fully_connected(fc , label.get_shape().as_list()[-1],
                                             scope = 'full')
                end_points['Logits'] = logits
        
    if Distill is not None:
        with tf.variable_scope('Teacher'):
            with tcf.arg_scope(WResNet_arg_scope_teacher()):
                depth = 40; widen_factor = 2 # teacher
                nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
                n = (depth-4)//6
                stride = [1,2,2]
                
                tch = tcl.conv2d(image, nChannels[0], [3,3], 1, scope='conv0')
                tf.add_to_collection('feat', tch)
                for i in range(3):            
                    tch = NetworkBlock(tch, n, nChannels[1+i], stride[i], name = 'WResblock%d'%i)
                    tf.add_to_collection('feat', tch)
                tch = tcl.batch_norm(tch,scope='bn0')
                tch = tf.nn.relu(tch)

                fc = tf.reduce_mean(tch, [1,2])
                logits_tch = tcl.fully_connected(fc , label.get_shape().as_list()[-1], 
                                                 biases_initializer = tf.zeros_initializer(),
                                                 scope = 'full')
                end_points['Logits_tch'] = logits_tch
                        
        with tcf.arg_scope([tcl.conv2d, tcl.fully_connected, tcl.batch_norm], trainable = True):
            with tcf.arg_scope([tcl.dropout, tcl.batch_norm], is_training = is_training):
                with tf.variable_scope('Distillation'):
                    feats = tf.get_collection('feat')
                    student_feats = feats[1:len(feats)//2]
                    teacher_feats = feats[len(feats)//2+1:]
            
                    if Distill == 'AT':
                       tf.add_to_collection('dist', Multiple.Attention_transfer(student_feats, teacher_feats))
                    elif Distill == 'Soft_logits':
                        with tf.variable_scope('KD'):
                            T = 4
                            Dist_loss = tf.reduce_mean(tf.reduce_sum( tf.nn.softmax(logits_tch/T)*(tf.nn.log_softmax(logits_tch/T)-tf.nn.log_softmax(logits/T)),1 ))
                        tf.add_to_collection('dist', Dist_loss)
                    elif Distill == 'FitNet':
                        tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))
                    elif Distill == 'VID-I':
                        tf.add_to_collection('dist', Dist.VID(student_feats, teacher_feats))
            
    return end_points

def WResNet_arg_scope(weight_decay=0.0005):
    with tcf.arg_scope([tcl.conv2d, tcl.fully_connected], 
                       weights_regularizer=tcl.l2_regularizer(weight_decay),
                       biases_initializer=None, activation_fn = None):
        with tcf.arg_scope([tcl.batch_norm], scale = False, center = True, activation_fn=None, 
                           variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'BN_collection']) as arg_sc:
            return arg_sc
            
def WResNet_arg_scope_teacher(weight_decay=0.0005):
    with tcf.arg_scope([tcl.conv2d, tcl.fully_connected], weights_regularizer = None, trainable = False,
                       variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']):
        with tcf.arg_scope([tcl.batch_norm], param_regularizers=None, trainable = False, is_training = False,
                           variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']) as arg_sc:
            return arg_sc
            
