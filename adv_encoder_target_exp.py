from mdt_cifar10_train import train_adv_encoder, adv_loss, create_train_op
from MDT_model import make_vgg16_model
from adv_net import adv_net, sym_adv_net, adv_net_mask, adv_train_net, adv_target_net, adv_target_net2, adv_target_net3
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

from cleverhans.utils_tf import model_eval, tf_model_load, model_eval_each_class
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
tf.app.flags.DEFINE_integer('target', 0,
                            """attack target""")
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mdt_cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

def adv_net_exp(data_dir, checkpoint_dir, train_mode,
        train_dir='./tmp/cifar10_train_adv_encoder', batch_size=128,
        data_aug=False,clip_norm=1.5, target=0, lr=0.0001):
    
    # sess get setting
    sess = tf.Session()

    model = make_vgg16_model(name = 'vgg16_eval_mode', eval_mode=True)

    # create mode feed
    train_feed = mode_feed(sess, True)
    eval_feed = mode_feed(sess, False)

    # train model
    if train_mode:
        # set input and get logits
        data_norm = False
        images, labels = mdt_cifar10_input.inputs(False, data_dir, batch_size,
                                                  data_aug, data_norm)

        labels = tf.constant(target,dtype=tf.int64, shape=(batch_size,))

        # dis_loss, output_images = adv_net(images)
        dis_loss, output_images = adv_target_net(images, clip_norm)

        logits = model(output_images)

        # attack seeting
        # c = 0.005
        c=1
        confidence = 0
        target = True

        # define model loss
        loss = adv_loss(dis_loss, logits, labels, target, confidence, c)

        global_step = tf.train.get_or_create_global_step()

        # train setting
        nb_epochs = 100
        lr = 0.0001
        # decay_rate = 0.99
        # decay_epochs = 1
        # decay_steps = decay_epochs*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//batch_size
        # lr = tf.train.exponential_decay(initial_lr,
        #                                 global_step,
        #                                 decay_steps,
        #                                 decay_rate,
        #                                 staircase=True)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)

        # define train variables
        adv_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          "adv_encoder")
        train_op = create_train_op(loss, global_step, adv_variables, opt)

        # ini all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)  

        # restore pre variables
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        var_info = tf.train.list_variables(ckpt.model_checkpoint_path)
        # print(var_info)
        var_name = [v[0] for v in var_info]

        restore_map = {variable.op.name:variable for variable in tf.global_variables()
                           if variable.op.name in var_name}
        # print(restore_map) 
        saver = tf.train.Saver(restore_map)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        #intialize global steps
        sess.run(global_step.initializer)

        # print(adv_variables)
        train_adv_encoder(sess, logits, loss, labels, train_op, train_dir, batch_size,
                          eval_feed, nb_epochs)

        sess.close()

    else:
        # define dataset format
        img_rows = 32
        img_cols = 32
        channels = 3
        nb_classes = 10

        # fetch data
        cifar10_data.maybe_download_and_return_python(data_dir)
        X, Y = mdt_cifar10_input.numpy_input(True, data_dir)

        Y = np.zeros_like(Y)
        Y[:] = target
        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))

        # dis_loss, output_images = adv_net(images)
        dis_loss, output_images = adv_target_net(x, clip_norm)

        logits = model(output_images)

        # restore trained model 
        if not checkpoint_load(sess, train_dir):
            return False
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(train_dir)
        # saver.restore(sess, ckpt.model_checkpoint_path)

        # create one-hot Y
        one_hot_Y = to_categorical(Y, nb_classes)

        # eval model accuracy
        accuracy = model_eval(sess, x, y, logits, X, one_hot_Y,
                              feed=eval_feed,
                              args={'batch_size': batch_size})
        print('model accuracy: {0}'.format(accuracy))

        sta_time = time.time()
        adv_imgs = adv_generate(sess, output_images, x, X, eval_feed, batch_size)
        end_time = time.time()
        duration = end_time - sta_time
        print('adv crafting time: {0}'.format(duration))

        #eval adv's l2 distance
        l2_dis = calculate_l2_dis(X/255, adv_imgs/255)
        print('adversarial examples\' mean l2 distance: {0}'.format(l2_dis))
        adv_imgs = np.around(adv_imgs).astype(int)
        # compare_show(X[9], adv_imgs[9])
        compare_show(X[16], adv_imgs[16])
        import matplotlib
        matplotlib.image.imsave('i_{0}_target_{1}.png'.format(FLAGS.i,FLAGS.target), adv_imgs[16])
        # matplotlib.image.imsave('i_{0}.png'.format(FLAGS.i), X[16])


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
    # print(feed)
    return feed

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
        # print(adv_imgs.shape)

    return adv_imgs


def main(argv=None):
    cifar10_data.maybe_download_and_extract_binary(FLAGS.data_dir)
    # for i in range(14):
    #     for target in range(3):
    #         train_dir='./tmp/cifar10_train_adv_encoder_'+str(i)+'_target_'+str(target)+'/'
    #         adv_net_exp(FLAGS.data_dir, FLAGS.checkpoint_dir, FLAGS.train_mode,
    #             train_dir, batch_size=FLAGS.batch_size, clip_norm=3-i*0.2, target=target)
    #         tf.reset_default_graph()

    i = FLAGS.i
    target= FLAGS.target
    lr = 0.0001 + i*0.00001
    train_dir='./tmp/cifar10_train_adv_encoder_'+str(i)+'_target_'+str(target)+'/'
    adv_net_exp(FLAGS.data_dir, FLAGS.checkpoint_dir, FLAGS.train_mode,
        train_dir, batch_size=FLAGS.batch_size, clip_norm=3-i*0.2, target=target, lr=lr)

    # i = 6
    # for target in range(7,10):
    #     # target=0
    #     lr = 0.0001 + i*0.00001
    #     train_dir='./tmp/cifar10_train_adv_encoder_'+str(i)+'_target_'+str(target)+'/'
    #     adv_net_exp(FLAGS.data_dir, FLAGS.checkpoint_dir, FLAGS.train_mode,
    #         train_dir, batch_size=FLAGS.batch_size, clip_norm=3-i*0.2, target=target, lr=lr)
    #     tf.reset_default_graph()

if __name__ == '__main__':
    tf.app.run()
