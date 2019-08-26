import tensorflow as tf

from tensorflow import ConfigProto

import time, os
import scipy.io as sio
import numpy as np
from random import shuffle

from nets import nets_factory
from dataloader import Dataloader
import op_util

home_path = os.path.dirname(os.path.abspath(__file__))

tf.app.flags.DEFINE_string('train_dir', 'test',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('Distillation', None,
                           'Distillation method : Soft_logits, FitNet, AT, FSP, DML, KD-SVD, FT, AB, RKD')
tf.app.flags.DEFINE_string('dataset', 'cifar100',
                           'Distillation method : cifar100, TinyImageNet, CUB200')
tf.app.flags.DEFINE_string('model_name', 'WResNet',
                           'Distillation method : ResNet, WResNet')
tf.app.flags.DEFINE_string('main_scope', 'Teacher',
                           'networ`s scope')
FLAGS = tf.app.flags.FLAGS
def main(_):
    ### define path and hyper-parameter
    Learning_rate =1e-1

    batch_size = 128
    val_batch_size = 200
    train_epoch = 200
    weight_decay = 5e-4

    should_log          = 100
    save_summaries_secs = 20
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_num = '0'

    if FLAGS.Distillation == 'None':
        FLAGS.Distillation = None
        
    train_images, train_labels, val_images, val_labels, pre_processing, teacher = Dataloader(FLAGS.dataset, home_path, FLAGS.model_name)
    num_label = int(np.max(train_labels)+1)

    rate = 1.
    if rate < 1:
        def slice(x, rate):
            return x[:int(x.shape[0]*rate)]
        idxes = np.hstack([slice(np.where(train_labels == i)[0], rate) for i in range(num_label)])
        train_images = train_images[idxes]
        train_labels = train_labels[idxes]
        
    dataset_len, *image_size = train_images.shape

    with tf.Graph().as_default() as graph:
        # make placeholder for inputs
        image_ph = tf.placeholder(tf.uint8, [None]+image_size)
        label_ph = tf.placeholder(tf.int32, [None])
        
        is_training_ph = tf.placeholder(tf.bool,[])
        
        # pre-processing
        image = pre_processing(image_ph, is_training_ph)
        label = tf.contrib.layers.one_hot_encoding(label_ph, num_label, on_value=1.0)
     
        # make global step
        global_step = tf.train.create_global_step()
        epoch = tf.floor_div(tf.cast(global_step, tf.float32)*batch_size, dataset_len)
        max_number_of_steps = int(dataset_len*train_epoch)//batch_size+1

        # make learning rate scheduler
        LR = learning_rate_scheduler(Learning_rate, [epoch, train_epoch], [0.3, 0.6, 0.8], 0.2)
        
        ## load Net
        class_loss, accuracy = MODEL(FLAGS.model_name, FLAGS.main_scope, weight_decay, image, label, is_training_ph, Distillation = FLAGS.Distillation)
        
        #make training operator
        train_op = op_util.Optimizer_w_Distillation(class_loss, LR, epoch, global_step, FLAGS.Distillation)
        
        ## collect summary ops for plotting in tensorboard
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
        
        ## make placeholder and summary op for training and validation results
        train_acc_place = tf.placeholder(dtype=tf.float32)
        val_acc_place   = tf.placeholder(dtype=tf.float32)
        val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
                       tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
        val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
        
        ## start training
        train_writer = tf.summary.FileWriter('%s'%FLAGS.train_dir,graph,flush_secs=save_summaries_secs)
        config = ConfigProto()
        config.gpu_options.visible_device_list = gpu_num
        config.gpu_options.allow_growth=True
        
        val_itr = len(val_labels)//val_batch_size
        logs = {'training_acc' : [], 'validation_acc' : []}
        with tf.Session(config=config) as sess:
            if FLAGS.Distillation is not None:
                global_variables  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                n = 0
                for v in global_variables:
                    if teacher.get(v.name[:-2]) is not None:
                        v._initial_value = tf.constant(teacher[v.name[:-2]].reshape(*v.get_shape().as_list()))
                        v._initializer_op = tf.assign(v._variable,v._initial_value,name= v.name[:-2]+'/Assign').op
                        n += 1
                print ('%d Teacher params assigned'%n)
            sess.run(tf.global_variables_initializer())
                
            sum_train_accuracy = []; time_elapsed = []; total_loss = []
            idx = list(range(train_labels.shape[0]))
            shuffle(idx)
            epoch_ = 0
            
            for step in range(max_number_of_steps):
                start_time = time.time()
                
                ## feed data
                tl, log, train_acc = sess.run([train_op, summary_op, accuracy],
                                              feed_dict = {image_ph : train_images[idx[:batch_size]],
                                                           label_ph : np.squeeze(train_labels[idx[:batch_size]]),
                                                           is_training_ph : True})
                time_elapsed.append( time.time() - start_time )
                total_loss.append(tl)
                sum_train_accuracy.append(train_acc)
                idx[:batch_size] = []
                if len(idx) < batch_size:
                    idx_ = list(range(train_labels.shape[0]))
                    shuffle(idx_)
                    idx += idx_
                
                step += 1
                if (step*batch_size)//dataset_len>=epoch_:
                    ## do validation
                    sum_val_accuracy = []
                    for i in range(val_itr):
                        acc = sess.run(accuracy, feed_dict = {image_ph : val_images[i*val_batch_size:(i+1)*val_batch_size],
                                                              label_ph : np.squeeze(val_labels[i*val_batch_size:(i+1)*val_batch_size]),
                                                              is_training_ph : False})
                        sum_val_accuracy.append(acc)
                        
                    sum_train_accuracy = np.mean(sum_train_accuracy)*100 if (step*batch_size)//dataset_len>init_epoch else 1.
                    sum_val_accuracy= np.mean(sum_val_accuracy)*100
                    tf.logging.info('Epoch %s Step %s - train_Accuracy : %.2f%%  val_Accuracy : %.2f%%'
                                    %(str(epoch_).rjust(3, '0'), str(step).rjust(6, '0'), 
                                    sum_train_accuracy, sum_val_accuracy))

                    result_log = sess.run(val_summary_op, feed_dict={train_acc_place : sum_train_accuracy,
                                                                     val_acc_place   : sum_val_accuracy   })
                    logs['training_acc'].append(sum_train_accuracy)
                    logs['validation_acc'].append(sum_val_accuracy)
    
                    if step == max_number_of_steps:
                        train_writer.add_summary(result_log, train_epoch)
                    else:
                        train_writer.add_summary(result_log, epoch_)
                    sum_train_accuracy = []

                    epoch_ += 1
                    
                    variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
                    
                if step % should_log == 0:
                    tf.logging.info('global step %s: loss = %.4f (%.3f sec/step)',str(step).rjust(6, '0'), np.mean(total_loss), np.mean(time_elapsed))
                    train_writer.add_summary(log, step)
                    time_elapsed = []
                    total_loss = []
                
                
                elif (step*batch_size) % dataset_len == 0:
                    train_writer.add_summary(log, step)

            ## save variables to use for something
            var = {}
            variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
            for v in variables:
                if v.name.split('/')[0] == FLAGS.main_scope:
                    var[v.name[:-2]] = sess.run(v)
            
            sio.savemat(FLAGS.train_dir + '/train_params.mat',var)
            sio.savemat(FLAGS.train_dir + '/log.mat',logs)
            
            ## close all
            tf.logging.info('Finished training! Saving model to disk.')
            train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
            train_writer.close()

def MODEL(model_name, scope, weight_decay, image, label, is_training, Distillation):
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay)
    end_points = network_fn(image, label, scope, is_training=is_training, Distill=Distillation)

    loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
    if Distillation == 'DML':
        tf.add_to_collection('teacher_class_loss',tf.losses.softmax_cross_entropy(label,end_points['Logits_tch']))
    accuracy = tf.contrib.metrics.accuracy(tf.cast(tf.argmax(end_points['Logits'], 1), tf.int32), tf.cast(tf.argmax(label, 1),tf.int32))
    return loss, accuracy
    
def learning_rate_scheduler(Learning_rate, epochs, decay_point, decay_rate):
    with tf.variable_scope('learning_rate_scheduler'):
        e, te = epochs
        for i, dp in enumerate(decay_point):
            Learning_rate = tf.cond(tf.greater_equal(e, int(te*dp)), lambda : Learning_rate*decay_rate, 
                                                                     lambda : Learning_rate)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate

if __name__ == '__main__':
    tf.app.run()

