import tensorflow as tf
import numpy as np

def VID(student_feature_maps, teacher_feature_maps):
    with tf.variable_scope('VID'):
        Distillation_loss = []
        for i, (sfm, tfm) in enumerate(zip(student_feature_maps[::-1], teacher_feature_maps[::-1])):
            Distillation_loss.append(0.)
            
        Distillation_loss =  tf.add_n(Distillation_loss)
        return Distillation_loss
