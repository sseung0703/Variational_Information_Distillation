import tensorflow as tf
import scipy.io as sio    
import glob

def Dataloader(name, home_path, model_name):
    if name == 'cifar100':
        return Cifar100(home_path, model_name)
    elif name == 'cifar10':
        return Cifar10(home_path, model_name)

def Cifar10(home_path, model_name):
    from tensorflow.keras.datasets.cifar10 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    teacher = sio.loadmat(home_path + '/pre_trained/ResNet40-2_cifar10.mat')\
              if len(glob.glob(home_path + '/pre_trained/ResNet40-2_cifar10.mat')) > 0 else None
    
    def pre_processing(image, is_training):
        with tf.variable_scope('preprocessing'):
            image = tf.cast(image, tf.float32)
            image = (image-128)/128
            def augmentation(image):
                image = tf.image.random_flip_left_right(image) # tf.__version__ > 1.10
                sz = tf.shape(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.random_crop(image,sz)
                return image
            image = tf.cond(is_training, lambda : augmentation(image), lambda : image)
        return image
    return train_images, train_labels, val_images, val_labels, pre_processing, teacher

def Cifar100(home_path, model_name):
    from tensorflow.keras.datasets.cifar100 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    teacher = sio.loadmat(home_path + '/pre_trained/%s_cifar100.mat'%model_name)\
              if len(glob.glob(home_path + '/pre_trained/%s_cifar100.mat'%model_name)) > 0 else None
    def pre_processing(image, is_training):
        with tf.variable_scope('preprocessing'):
            image = tf.cast(image, tf.float32)
            image = (image-128)/128
            def augmentation(image):
                image = tf.image.random_flip_left_right(image) # tf.__version__ > 1.10
                sz = tf.shape(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.random_crop(image,sz)
                return image
            image = tf.cond(is_training, lambda : augmentation(image), lambda : image)
        return image
    return train_images, train_labels, val_images, val_labels, pre_processing, teacher    
