import functools

import tensorflow as tf

from nets import WResNet

networks_map   = {
                 'WResNet':WResNet.WResNet,
                 }

arg_scopes_map = {
                  'WResNet':WResNet.WResNet_arg_scope,
                 }

def get_network_fn(name, weight_decay=5e-4):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, label, scope, is_training, Distill):
        with tf.contrib.framework.arg_scope(arg_scope):
            return func(images, label, scope = scope, is_training=is_training, Distill=Distill)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn

