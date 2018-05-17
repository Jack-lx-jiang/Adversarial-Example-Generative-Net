from mdt_cifar10_train import train_process, stand_loss, adv_net_loss
from MDT_model import make_mdt_model, make_cnn_drop_model, make_vgg16_model, make_vgg16_clipRelu_model, make_vgg16_clipRelu_ordering_exchange_model
from mdt_cifar10_eval_np import checkpoint_load, evaluate_uncertainty, calculate_l2_dis, compare_show
import tensorflow as tf
import cifar10_data
import mdt_cifar10_input
import matplotlib.pyplot as plt
import os
import pickle
import time

from cleverhans.utils_tf import model_eval, tf_model_load, model_eval_each_class
from cleverhans.utils import to_categorical
from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/cifar10_train',
                           """Directory where to load checkpoints """)
tf.app.flags.DEFINE_string('adversarial_dir', './tmp/cifar10_adv',
                           """Directory where to save adversarial examples """)


def mdt(model, data_dir, checkpoint_dir,
        train_dir='./tmp/cifar10_train',
        adversarial_dir='./tmp/cifar10_adv', batch_size=128,
        data_aug=False, data_norm=True):

    # train model
    if not tf.gfile.Exists(train_dir):
        # set input and get logits
        images, labels = mdt_cifar10_input.inputs(False, data_dir, batch_size,
                                                  data_aug, data_norm)

        labels = tf.cast(labels, tf.int64)
        # target = False
        # adv_output_layer = 'adv_bounddecoder6'
        # loss = adv_net_loss(images, model, labels, target, adv_output_layer, 0, 10)
        logits = model(images)
        loss = stand_loss(logits, labels)
        train_process(model, loss, images, label, train_dir, batch_size)
    
    # define dataset format
    img_rows = 32
    img_cols = 32
    channels = 3
    nb_classes = 10

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Get predict tensor
    pred = model(x)

    sess = tf.Session()

    if not checkpoint_load(sess, checkpoint_dir):
        return False

    # fetch data
    cifar10_data.maybe_download_and_return_python(data_dir)
    X, Y = mdt_cifar10_input.numpy_input(True, data_dir)

    # create one-hot Y
    one_hot_Y = to_categorical(Y, nb_classes)

    # create mode feed
    train_feed = mode_feed(sess, True)
    eval_feed = mode_feed(sess, False)

    fgsm_params = {'eps': 1,
                   'clip_min': 0.,
                   'clip_max': 255.}
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_probs(adv_x)
 
    # eval model accuracy
    class_accuracy, accuracy = model_eval_each_class(sess, x, y, pred, 10, X, one_hot_Y,
                          feed=eval_feed,
                          args={'batch_size': 128})
    print('model accuracy: {0}'.format(accuracy))

    for i in range(10):
        print('class {0} accuracy: {1}'.format(i, class_accuracy[i]))

    # eval model's accuacy in cw adversarial examples
    fgsm_accuracy = model_eval(sess, x, y, preds_adv, X, one_hot_Y,
                             feed=eval_feed,
                             args={'batch_size': 128})
    print('model fgsm_accuracy: {0}'.format(fgsm_accuracy))


    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}
                   

    X = X[:128]
    Y=one_hot_Y[:128]
    adv_feed = {x:X, y:one_hot_Y}
    adv_feed.update(eval_feed)
    sta = time.time()
    adv_X_ = sess.run(adv_x,feed_dict=adv_feed)
    end = time.time()
    duration = end - sta
    print('finished in {0} seconds'.format(duration))

    l2_dis = calculate_l2_dis(X/255, adv_X_/255)
    print('adversarial examples\' mean l2 distance: {0}'.format(l2_dis))


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


def main(argv=None):
    cifar10_data.maybe_download_and_extract_binary(FLAGS.data_dir)
    model = make_vgg16_model(name = 'vgg16_eval_model', eval_mode=True)
    # model = make_vgg16_model(name = 'vgg16_model')
    # model = make_cnn_drop_model(name = 'cnn_0.9_dropout_0.5_model')
    # model = make_vgg16_clipRelu_model(name='vgg16_clipReLu', eval_mode=True)
    # model = make_vgg16_clipRelu_ordering_exchange_model(name='vgg16_clipRelu_ordering_exchange')
    mdt(model, FLAGS.data_dir, FLAGS.checkpoint_dir,
        FLAGS.train_dir, FLAGS.adversarial_dir, FLAGS.batch_size)

if __name__ == '__main__':
    tf.app.run()
