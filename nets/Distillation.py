import tensorflow as tf
import numpy as np

def VID(student_feature_maps, teacher_feature_maps, l = 1e-1):
    with tf.variable_scope('VID'):
        Distillation_loss = []
        for i, (sfm, tfm) in enumerate(zip(student_feature_maps[::-1], teacher_feature_maps[::-1])):
            with tf.variable_scope('vid%d'%i):
                C = sfm.get_shape().as_list()[-1]

                sfm = tf.contrib.layers.flatten(sfm)
                tfm = tf.contrib.layers.flatten(tfm)
                
                alpha = tf.get_variable('alpha', [C], tf.float32, trainable = True, initializer = tf.constant_initializer(5.))
                stddev = tf.math.softplus(alpha)+1e-12
                mean   = tf.reduce_mean(sfm, 1, keepdims=True)


                vid_loss = l * tf.reduce_mean(tf.reduce_sum(tf.tf.log(stddev) + tf.square(tfm - mean)/(2*tf.square(stddev)),1))
                
                Distillation_loss.append(vid_loss)
            
        Distillation_loss =  tf.add_n(Distillation_loss)
        return Distillation_loss
