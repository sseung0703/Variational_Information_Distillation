import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def Optimizer_w_Distillation(class_loss, LR, global_step, Distillation):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables)-set(teacher_variables))
        
        # make optimizer w/ learning rate scheduler
        optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        if Distillation is None:
            total_loss = class_loss + tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients  = optimize.compute_gradients(total_loss, var_list = variables)
            
        elif Distillation == 'Soft_logits':
            total_loss = tf.add_n(tf.losses.get_regularization_losses()) + class_loss*0.7 + tf.get_collection('dist')[0]*0.3
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients  = optimize.compute_gradients(total_loss, var_list = variables)
            
        elif Distillation in {'AT'}:
            total_loss = class_loss + tf.add_n(tf.losses.get_regularization_losses() + tf.get_collection('dist')) 
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients  = optimize.compute_gradients(total_loss, var_list = variables)
            
        elif Distillation in {'VID-I'}:
            total_loss = class_loss*.1 + tf.add_n(tf.losses.get_regularization_losses())*.1 + tf.add_n(tf.get_collection('dist')) 
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients  = optimize.compute_gradients(total_loss, var_list = variables)
            with tf.variable_scope('clip_grad'):
                gradients = [(tf.clip_by_norm(g, 100), v) for i, (g, v) in enumerate(gradients)]
    
        # merge update operators and make train operator
        update_ops.append(optimize.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        return train_op

def Optimizer_w_Initializer(class_loss, LR, global_step):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables)-set(teacher_variables))
        
        # make optimizer w/ learning rate scheduler
        optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        # initialization and fine-tuning
        # in initialization phase, weight decay have to be turn-off which is not trained by distillation
        reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        distillation_loss = tf.get_collection('dist')[0]

        total_loss = class_loss + reg_loss
        tf.summary.scalar('loss/total_loss', total_loss)
        gradients  = optimize.compute_gradients(total_loss, var_list = variables)
        
        gradient_dist   = optimize.compute_gradients(distillation_loss, var_list = variables)
        gradient_wdecay = optimize.compute_gradients(reg_loss,          var_list = variables)
        with tf.variable_scope('clip_grad'):
            for i, (gw, gd) in enumerate(zip(gradient_wdecay, gradient_dist)):
                if gd[0] is not None and gw[0] is not None:
                    gradient_dist[i] = (gw[0] + gd[0], gd[1])

        # merge update operators and make train operator
        update_ops.append(optimize.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        
        update_ops_dist = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_dist.append(optimize.apply_gradients(gradient_dist, global_step=global_step))
        update_op_dist = tf.group(*update_ops_dist)
        train_op_dist = control_flow_ops.with_dependencies([update_op_dist], distillation_loss, name='train_op_dist')
        return train_op, train_op_dist
