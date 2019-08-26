import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def Optimizer_w_Distillation(class_loss, LR, epoch, global_step, Distillation):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables)-set(teacher_variables))
        
        # make optimizer w/ learning rate scheduler
        optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        # training main-task
        total_loss = class_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('loss/total_loss', total_loss)
        gradients  = optimize.compute_gradients(total_loss, var_list = variables)
        
                
        # merge update operators and make train operator
        update_ops.append(optimize.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        return train_op


