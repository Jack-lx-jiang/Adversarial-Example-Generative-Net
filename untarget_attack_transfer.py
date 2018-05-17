from mdt_cifar10_train import train_adv_encoder, adv_loss, create_train_op
from MDT_model import make_vgg16_clipRelu_model, make_mdt_model, make_vgg16_model, make_standard_model
from adv_net import adv_net, sym_adv_net, adv_net_mask, adv_train_net
from mdt_cifar10_eval_np import checkpoint_load, evaluate_uncertainty, calculate_l2_dis, compare_show
import tensorflow as tf
import cifar10_data
import mdt_cifar10_input
import matplotlib.pyplot as plt
import os
import pickle
import time
import numpy as np
import math

from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.utils import to_categorical
from cleverhans.attacks import CarliniWagnerL2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/cifar10_train',
                           """Directory where to load checkpoints """)
tf.app.flags.DEFINE_string('adversarial_dir', './tmp/cifar10_adv',
                           """Directory where to save adversarial examples """)
tf.app.flags.DEFINE_boolean('train_mode', True,
                            """Whether to train or eval.""")
tf.app.flags.DEFINE_integer('i', 0,
                            """index""")
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mdt_cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

def adv_net_exp(data_dir, adv_dir,
                target_model_dir='./tmp/cifar10_train_adv_encoder',
                clip_norm=1.5):
    
    # sess get setting
    sess = tf.Session()


    # define dataset format
    img_rows = 32
    img_cols = 32
    channels = 3
    nb_classes = 10

    # fetch data
    cifar10_data.maybe_download_and_return_python(data_dir)
    X, Y = mdt_cifar10_input.numpy_input(True, data_dir)
    
    # create one-hot Y
    one_hot_Y = to_categorical(Y, nb_classes)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    model = make_vgg16_clipRelu_model(name = 'vgg16_clipRelu_eval_mode', eval_mode=True)
    
    eval_feed = mode_feed(sess, False)
    # Get predict tensor
    pred = model(x)
    if not checkpoint_load(sess, target_model_dir):
        return False

    # eval model accuracy
    accuracy = model_eval(sess, x, y, pred, X, one_hot_Y,
                          feed = eval_feed,
                          args={'batch_size': 128})
    print('model accuracy: {0}'.format(accuracy))

    dis_loss, output_images = adv_train_net(x, clip_norm)

    logits = model(output_images)

    # restore adv variables
    ckpt = tf.train.get_checkpoint_state(adv_dir)
    # define adv variables
    adv_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "adv_encoder")
    saver = tf.train.Saver(adv_variables)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # eval adv accuracy
    accuracy = model_eval(sess, x, y, logits, X, one_hot_Y,
                          feed = eval_feed,
                          args={'batch_size': 128})
    print('transfer rate: {0}'.format(accuracy))


    # universal adversarial examples
    adv_imgs = adv_generate(sess, output_images, x, X, None, 128)
    mean_dif = adv_imgs[1]-X[1]
    print('mean dif\'s size: {0}'.format(mean_dif.shape))
    universal_adv_X = X+mean_dif
    # eval universal adv accuracy
    accuracy = model_eval(sess, x, y, pred, universal_adv_X, one_hot_Y,
                          feed = eval_feed,
                          args={'batch_size': 128})
    print('universal adv transfer rate: {0}'.format(accuracy))

def adv_generate(sess, output_images, x, X, feed, batch_size):
    adv_imgs = []
    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X)) / batch_size))
        assert nb_batches * batch_size >= len(X)

        X_cur = np.zeros((batch_size,) + X.shape[1:],
                         dtype=X.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X[start:end]
            feed_dict = {x: X_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_output_images = output_images.eval(feed_dict=feed_dict)

            for cur in range(cur_batch_size):
                adv_imgs.append(cur_output_images[cur])

        assert end >= len(X)

        # Divide by number of examples to get final value
        adv_imgs = np.array(adv_imgs)

    return adv_imgs

def mode_feed(sess, mode):
    dropout = tf.get_collection('dropout')
    bn = tf.get_collection('bn_mode')
    if mode:
        # deploy dropout prob
        dropout_train_setting = sess.run(dropout)
        feed = dict(zip(dropout,dropout_train_setting))
        bn_train = {b:True for b in bn}
        feed.update(bn_train)
    else:
        dropout_eval_setting = [1.0 for i in range(len(dropout))]
        feed = dict(zip(dropout,dropout_eval_setting))
        bn_eval = {b:False for b in bn}
        feed.update(bn_eval)
    return feed

def main(argv=None):
    cifar10_data.maybe_download_and_extract_binary(FLAGS.data_dir)
    # adv_dir='./tmp/cifar10_train_adv_encoderncoder_clip_norm1.5'
    adv_dir='./tmp/cifar10_train_adv_encoder_3.0_ clip_norm'
    target_model_dir = './tmp/vgg16_clip_ReLu/'
    adv_net_exp(FLAGS.data_dir, adv_dir, 
                target_model_dir, clip_norm=3.0)

if __name__ == '__main__':
    tf.app.run()
