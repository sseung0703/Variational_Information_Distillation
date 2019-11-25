import subprocess
import tensorflow as tf
import glob
import scipy.io as sio
import numpy as np


train_acc_place = tf.placeholder(dtype=tf.float32)
val_acc_place   = tf.placeholder(dtype=tf.float32)
val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
               tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
def get_avg_plot(sess, base_path):
    pathes = glob.glob(base_path[:-len(base_path.split('/')[-1])] + '*')
    training_acc   = []
    validation_acc = []
    for path in pathes:
        logs = sio.loadmat(path + '/log.mat')
        training_acc.append(logs['training_acc'])
        validation_acc.append(logs['validation_acc'])
    training_acc   = np.mean(np.vstack(training_acc),0)
    validation_acc = np.mean(np.vstack(validation_acc),0)

    train_writer = tf.summary.FileWriter(base_path[:-len(base_path.split('/')[-1])] + 'average',flush_secs=1)
    for i, (train_acc, val_acc) in enumerate(zip(training_acc,validation_acc)):
        result_log = sess.run(val_summary_op, feed_dict={train_acc_place : train_acc,
                                                         val_acc_place   : val_acc   })
        train_writer.add_summary(result_log, i)
    train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
    train_writer.close()
    
conf = 1
with tf.Session(config=config) as sess:
    if conf == 0:
        for d in ['VID-I']:
            base_path = '/home/cvip/Documents/VID/RTD/full/16-1/%s_/%s'%(d,d)
            for i in range(3):
                subprocess.call('python Variational_Information_Distillation/train_w_distill.py '
                           +'--train_dir=%s%d '%(base_path,i)
                           +'--model_name=WResNet '
                           +'--main_scope=Student '
                           +'--dataset=cifar10 '
                           +'--Distillation=%s '%d
                           +'--rate=.full ',
                           shell=True)
        get_avg_plot(sess, base_path)

    if conf == 1:
        for r in ['.02', '.10', '.20']:
            d = 'VID-I'
            base_path = '/home/cvip/Documents/VID/RTD/%s/16-1/%s/%s'%(r[1:],d,d)
            for i in range(3):
                subprocess.call('python Variational_Information_Distillation/train_w_distill.py '
                           +'--train_dir=%s%d '%(base_path,i)
                           +'--model_name=WResNet '
                           +'--main_scope=Student '
                           +'--dataset=cifar10 '
                           +'--Distillation=%s '%d
                           +'--rate=%s '%r,
                           shell=True)
        get_avg_plot(sess, base_path)
