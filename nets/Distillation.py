import tensorflow as tf
import numpy as np


def VID(student_feature_maps, teacher_feature_maps, l = 1e-1):
    with tf.variable_scope('VID'):
        Distillation_loss = []
        for i, (sfm, tfm) in enumerate(zip(student_feature_maps, teacher_feature_maps)):
            with tf.variable_scope('vid%d'%i):
                C = tfm.get_shape().as_list()[-1]
                if len(tfm.get_shape().as_list()) > 2:
                    for i in range(3):
                        sfm = tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(sfm, C if i == 2 else C*2, scope = 'fc%d'%i),
                                                           activation_fn = tf.nn.relu, scope = 'bn%d'%i)
                
                alpha = tf.get_variable('alpha', [1,1,1,C], tf.float32, trainable = True, initializer = tf.constant_initializer(5.))
                var   = tf.math.softplus(alpha)+1e-3
                vid_loss = tf.reduce_mean(tf.log(var) + tf.square(tfm - sfm)/var)/2
                
                Distillation_loss.append(vid_loss)
            
        Distillation_loss =  tf.add_n(Distillation_loss)
        return Distillation_loss
