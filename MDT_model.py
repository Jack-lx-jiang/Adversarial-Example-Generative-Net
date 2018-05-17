"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


class MDT_Model(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape, name=None, name_count_start=0):
        super(MDT_Model, self).__init__()
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__
        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if layer.name is not None:
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i-name_count_start)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):

    def __init__(self, num_hid, l2=0, name=None):
        self.num_hid, self.l2, self.name = num_hid, l2, name

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        with tf.variable_scope(self.name) as scope:
            init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                       keep_dims=True))
            self.W = tf.Variable(init, name='Weight')
            if(self.l2 != 0):
                weight_loss = tf.multiply(tf.nn.l2_loss(
                    self.W), self.l2, name='weight_loss')
                tf.add_to_collection('losses', weight_loss)
            self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'), name='Bias')

    def fprop(self, x):
        return tf.matmul(x, self.W, name=self.name) + self.b


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding,
                 l2=0, name=None):
        self.output_channels, self.kernel_shape, self.strides, self.padding, self.l2, self.name = output_channels, kernel_shape, strides, padding, l2, name

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        with tf.variable_scope(self.name) as scope:
            init = tf.random_normal(kernel_shape, dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                       axis=(0, 1, 2)))
            self.kernels = tf.Variable(init, name='Weight')
            # tf.add_to_collection('test', self.kernels)
            if(self.l2 != 0):
                weight_loss = tf.multiply(tf.nn.l2_loss(
                    self.kernels), self.l2, name='weight_loss')
                tf.add_to_collection('losses', weight_loss)
            self.b = tf.Variable(
                np.zeros((self.output_channels,)).astype('float32'),name='Bias')
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            dummy_output = self.fprop(dummy_batch)
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = batch_size
            self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding, name=self.name) + self.b


class ReLU(Layer):

    def __init__(self, name=None):
        self.name = None

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x, self.name)


class Softmax(Layer):

    def __init__(self, name=None):
        self.name = name

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self, name=None):
        self.name = name

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [shape[0], output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


class MaxPool(Layer):
    def __init__(self, kernel_shape, strides, padding, name=None):
        self.kernel_shape, self.strides, self.padding, self.name = kernel_shape, strides, padding, name

    def set_input_shape(self, input_shape):
        input_shape = list(input_shape)
        batch_size = input_shape[0]
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.max_pool(x, (1,)+tuple(self.kernel_shape)+(1,), (1,)+tuple(self.strides)+(1,),
                              self.padding, name=self.name)


class LRN(Layer):
    def __init__(self, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None):
        self.depth_radius, self.bias, self.alpha, self.beta, self.name = depth_radius, bias, alpha, beta, name

    def set_input_shape(self, input_shape):
        self.output_shape = input_shape

    def fprop(self, x):
        return tf.nn.lrn(x, self.depth_radius, self.bias, self.alpha, self.beta, self.name)


class Dropout(Layer):
    def __init__(self, keep_prob, seed=None, name=None):
        self.keep_prob, self.seed, self.name = keep_prob, seed, name

    def set_input_shape(self, input_shape):
        with tf.variable_scope(self.name) as scope:
            self.prob = tf.placeholder_with_default(self.keep_prob, shape=(),
                                                    name='dropout_prob')
            tf.add_to_collection('dropout', self.prob)
        self.output_shape = input_shape

    def fprop(self, x):
        return tf.nn.dropout(x, self.prob, seed=self.seed, name=self.name)


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99, epsilon=0.001, name=None):
        self.momentum, self.epsilon, self.name = momentum, epsilon, name
    
    def set_input_shape(self, input_shape)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          :
        with tf.variable_scope(self.name) as scope:
            # self.beta_initializer = tf.zeros_initializer()
            # self.gamma_initializer = tf.ones_initializer()
            # self.moving_mean_initializer = tf.zeros_initializer()
            # self.moving_variance_initializer = tf.ones_initializer()
            # self.beta = tf.Variable(tf.constant(0.0, shape=[input_shape[-1]]), name='beta', trainable=True)
            # self.gamma = tf.Variable(tf.constant(1.0, shape=[input_shape[-1]]), name='gamma', trainable=True)

            self.training = tf.placeholder_with_default(True, shape=(),
                                                        name='bn_istraining')

            tf.add_to_collection('bn_mode', self.training)

        self.reuse = False
        
        self.output_shape = input_shape

    def fprop(self, x):

        bn = tf.layers.batch_normalization(x, momentum=self.momentum,
                                             epsilon=self.epsilon,
                                             name=self.name,
                                             reuse=self.reuse,
                                             trainable=True,
                                             training=self.training)
                                     
        self.reuse = True
        return bn

class ClipReLu(Layer):
    def __init__(self, bounder=6, name=None):
        self.bounder, self.name = bounder, name

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.clip_by_value(x, -self.bounder, self.bounder, self.name)

class LinearToConv(Layer):

    def __init__(self, shape, name=None):
        self.shape, self.name = shape, name

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = list(self.shape).copy()
        self.output_shape[0] = shape[0]

    def fprop(self, x):
        return tf.reshape(x, self.shape)

class Norm(Layer):
    def __init__(self, name=None):
        self.name = name

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        mean, var = tf.nn.moments(x, axes=tuple(range(1,len(x.shape))), keep_dims=True)
        return (x-mean)/tf.sqrt(var)

class BoundEcoder(Layer):
    def __init__(self, min_bound=0, max_bound=255, name=None):
        self.min_bound, self.max_bound, self.name = min_bound, max_bound, name
    
    def set_input_shape(self, shape):
        self.output_shape = shape

    def fprop(self, x):
        imgs = (x-self.min_bound)/(self.max_bound-self.min_bound)
        return imgs

class BoundDecoder(Layer):
    def __init__(self, min_bound=0, max_bound=255, name=None):
        self.min_bound, self.max_bound, self.name = min_bound, max_bound, name
    
    def set_input_shape(self, shape):
        self.output_shape = shape

    def fprop(self, x):
        imgs = x*(self.max_bound-self.min_bound)+self.min_bound
        return imgs

class Tanh(Layer):

    def __init__(self, name=None):
        self.name = None

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.tanh(x, self.name)

class Sigmoid(Layer):

    def __init__(self, name=None):
        self.name = None

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.sigmoid(x, self.name)

class Conv2D_Transpose(Layer):

    def __init__(self, output_channels, kernel_shape, output_shape_t, strides, padding,
                 l2=0, name=None):
        self.output_channels, self.kernel_shape, self.output_shape_t, self.strides, self.padding, self.l2, self.name = output_channels, kernel_shape, output_shape_t, strides, padding, l2, name

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        with tf.variable_scope(self.name) as scope:
            init = tf.random_normal(kernel_shape, dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                       axis=(0, 1, 2)))
            self.kernels = tf.Variable(init, name='Weight')
            # tf.add_to_collection('test', self.kernels)
            if(self.l2 != 0):
                weight_loss = tf.multiply(tf.nn.l2_loss(
                    self.kernels), self.l2, name='weight_loss')
                tf.add_to_collection('losses', weight_loss)
            self.b = tf.Variable(
                np.zeros((self.output_channels,)).astype('float32'),name='Bias')
            self.output_shape = list(self.output_shape_t).copy()
            self.output_shape[0] = batch_size

    def fprop(self, x):
        return tf.nn.conv2d_transpose(x, self.kernels, list(self.output_shape_t), (1,) + tuple(self.strides) + (1,),
                            self.padding, name=self.name) + self.b


def make_standard_model(name=None, eval_mode=False):
    l2 = 0.004
    layers = [Conv2D(64, (5, 5), (1, 1), 'SAME', l2),
              ReLU(),
              MaxPool((3, 3), (2, 2), 'SAME'),
              LRN(4, 1, 0.001 / 9.0, 0.75),
              Conv2D(64, (5, 5), (1, 1), 'SAME', l2),
              ReLU(),
              LRN(4, 1, 0.001 / 9.0, 0.75),
              MaxPool((3, 3), (2, 2), 'SAME'),
              Flatten(),
              Linear(384, l2),
              ReLU(),
              Linear(192, l2),
              ReLU(),
              Linear(10, l2)]
    # model = MDT_Model(layers, (None, 32, 32, 3))
    if eval_mode:
        eval_layer = [Norm('input_norm_layer')]
        model = MDT_Model(eval_layer+layers, (None, 32, 32, 3), name, len(eval_layer))
    else:
        model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model

def make_mdt_model(name = None, eval_mode=False):
    l2 = 0
    dropout = 1.0
    layers = [Conv2D(64, (5, 5), (1, 1), 'SAME', l2),
              ReLU(),
              MaxPool((3, 3), (2, 2), 'SAME'),
              LRN(4, 1, 0.001 / 9.0, 0.75),
              Conv2D(64, (5, 5), (1, 1), 'SAME', l2),
              ReLU(),
              LRN(4, 1, 0.001 / 9.0, 0.75),
              MaxPool((3, 3), (2, 2), 'SAME'),
              Flatten(),
              Linear(384, l2),
              Dropout(dropout),
              ReLU(),
              Linear(192, l2),
              Dropout(dropout),
              ReLU(),
              Linear(10, l2)]
    if eval_mode:
        eval_layer = [Norm('input_norm_layer')]
        model = MDT_Model(eval_layer+layers, (None, 32, 32, 3), name, len(eval_layer))
    else:
        model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model

def make_cnn_drop_model(name=None):
    l2 = 0
    dropout1 = 0.9
    dropout2 = 0.5
    layers = [Dropout(dropout1),
              Conv2D(64, (5, 5), (1, 1), 'SAME', l2),
              Dropout(dropout1),
              ReLU(),
              MaxPool((3, 3), (2, 2), 'SAME'),
              # LRN(4, 1, 0.001 / 9.0, 0.75),
              Conv2D(64, (5, 5), (1, 1), 'SAME', l2),
              Dropout(dropout1),
              ReLU(),
              # LRN(4, 1, 0.001 / 9.0, 0.75),
              MaxPool((3, 3), (2, 2), 'SAME'),
              Flatten(),
              Linear(384, l2),
              Dropout(dropout2),
              ReLU(),
              Linear(192, l2),
              Dropout(dropout2),
              ReLU(),
              Linear(10, l2)]
    model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model

def make_vgg16_model(name=None, eval_mode=False):

    l2 = 0.0005
    layers=[Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Dropout(0.5),
            Flatten(),
            Linear(512, l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.5),
            Linear(10)]
    if eval_mode:
        eval_layer = [Norm('input_norm_layer')]
        model = MDT_Model(eval_layer+layers, (None, 32, 32, 3), name, len(eval_layer))
    else:
        model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model


def make_vgg16_clipRelu_model(name=None, eval_mode=False):

    l2 = 0.0005
    layers=[Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Dropout(0.5),
            Flatten(),
            Linear(512, l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.5),
            Linear(10)]
            
    if eval_mode:
        eval_layer = [Norm('input_norm_layer')]
        model = MDT_Model(eval_layer+layers, (None, 32, 32, 3), name, len(eval_layer))
    else:
        model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model

def make_vgg16_clipRelu_maxpool_fix_model(name=None):

    l2 = 0.0005
    layers=[Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            BatchNormalization(),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            BatchNormalization(),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ClipReLu(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Dropout(0.5),
            Flatten(),
            Linear(512, l2),
            ClipReLu(),
            BatchNormalization(),
            Dropout(0.5),
            Linear(10)]
    model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model

def make_vgg16_clipRelu_ordering_exchange_model(name=None):

    l2 = 0.0005
    layers=[Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.3),
            Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            BatchNormalization(),
            ClipReLu(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Dropout(0.5),
            Flatten(),
            Linear(512, l2),
            BatchNormalization(),
            ClipReLu(),
            Dropout(0.5),
            Linear(10)]
    model = MDT_Model(layers, (None, 32, 32, 3), name)
    return model

def make_adv_ecoder_model(name=None):
    l2 = 0.0005
    # one Normalize Layer to be added, Also clip to bound operation
    adv_net=[BoundEcoder(0,255,'adv_boundecoder0'),
             Conv2D(32,(3,3),(1,1),'VALID', name='adv_conv'),
             Tanh(),
             Flatten('Flatten1'),
             Linear(128, name='adv_net_Linear2'),
             Tanh(),
             Linear(2700),
             Tanh(),
             LinearToConv([-1,30,30,3]),
             Conv2D_Transpose(3,(3,3),(128,32,32,3),(1,1),'VALID', name='adv_deconv'),
             Sigmoid(),
             # ReLU(),
             # Linear(6144, name='adv_net_Linear3'),
             # ReLU(),
             # Linear(3072, name='adv_net_Linear4'),
             # LinearToConv([-1,32,32,3], name='adv_net_LTC5'),
             BoundDecoder(0,255,'adv_bounddecoder6'),
             Norm(name='adv_net_Norm7')]
    layers=[Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(128, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(256, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.4),
            Conv2D(512, (3, 3), (1, 1), 'SAME', l2),
            ReLU(),
            BatchNormalization(),
            MaxPool((2, 2), (2, 2), 'SAME'),
            Dropout(0.5),
            Flatten(),
            Linear(512, l2),
            ReLU(),
            BatchNormalization(),
            Dropout(0.5),
            Linear(10)]
    model = MDT_Model(adv_net+layers, (None, 32, 32, 3), name, len(adv_net))
    return model