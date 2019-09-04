import tensorflow as tf

def FitNet(student, teacher):
    '''
     Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
     Fitnets:   Hints  for  thin  deep  nets.
     arXiv preprint arXiv:1412.6550, 2014.
    '''
    def Guided(source, target):
        with tf.variable_scope('Guided'):
            Ds = source.get_shape().as_list()[-1]
            Dt = target.get_shape().as_list()[-1]
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    target = tf.contrib.layers.fully_connected(target, Ds, biases_initializer = None, trainable=True, scope = 'fc')
            
            return tf.reduce_mean(tf.square(source-target))
    return tf.add_n([Guided(std, tch) for i, std, tch in zip(range(len(student)), student, teacher)])

def Attention_transfer(student, teacher, beta = 1e3):
    '''
     Zagoruyko, Sergey and Komodakis, Nikos.
     Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer.
     arXiv preprint arXiv:1612.03928, 2016.
    '''
    def Attention(source, target):
        with tf.variable_scope('Attention'):
            Qt = tf.reduce_mean(tf.square(source),-1)
            Qt = tf.nn.l2_normalize(Qt, [1,2])
            
            Qs = tf.reduce_mean(tf.square(target),-1)
            Qs = tf.nn.l2_normalize(Qs, [1,2])
            
            return tf.reduce_mean(tf.square(Qt-Qs))*beta/2
    return tf.add_n([Attention(std, tch) for std, tch in zip(student, teacher)])
    
