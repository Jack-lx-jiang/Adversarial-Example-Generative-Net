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
from cleverhans.attacks import CarliniWagnerL2

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('data_dir', './tmp/cifar10_data',
#                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_integer('batch_size', 128,
#                             """Number of images to process in a batch.""")

# tf.app.flags.DEFINE_string('train_dir', './tmp/cifar10_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
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

    # print(sess.run(bn_moving_vars))


    # create one-hot Y
    one_hot_Y = to_categorical(Y, nb_classes)

    # create mode feed
    train_feed = mode_feed(sess, True)
    eval_feed = mode_feed(sess, False)

    # craft cw adversarial examples
    if not os.path.exists(adversarial_dir):
        os.makedirs(adversarial_dir)
    cw_file = adversarial_dir+'/cw_adv'
    if os.path.isfile(cw_file):
        fr = open(cw_file, 'rb')
        cw_dict = pickle.load(fr)
        cw_adv = cw_dict['data']
        adv_ys = cw_dict['labels']
        assert cw_adv.shape[0] == adv_ys.shape[0]
        cw_setting = cw_dict['setting']
        print('settings of cw adversarial examples that have been loaded')
        print(cw_setting)
    else:
        print('crafting cw adversarial examples....')
        start_time = time.time()

        cw = CarliniWagnerL2(model, back='tf', sess=sess)
        num_for_test = 100
        adv_inputs = X[:num_for_test]
        yname = 'y'
        adv_ys = one_hot_Y[:num_for_test]

        cw_params = {'binary_search_steps': 5,
                     'confidence':0,
                     'max_iterations': 10000,
                     'learning_rate': 0.1,
                     'batch_size': 100,
                     'initial_const': 10,
                     'clip_min': 0,
                     'clip_max': 255}

        cw_setting = cw_params.copy()

        cw_params['feed'] = eval_feed
        cw_params[yname] = adv_ys

        cw_adv = cw.generate_np(adv_inputs,
                             **cw_params)
        cw_setting['model'] = model.name
        cw_dict = {'data':cw_adv, 'labels':adv_ys, 'setting':cw_setting}
        fw = open(cw_file, 'wb')
        pickle.dump(cw_dict, fw)

        end_time = time.time()
        duration = end_time - start_time
        print('finished in {0} seconds'.format(duration))

    # eval model accuracy
    class_accuracy, accuracy = model_eval_each_class(sess, x, y, pred, 10, X, one_hot_Y,
                          feed=eval_feed,
                          args={'batch_size': 128})
    print('model accuracy: {0}'.format(accuracy))

    for i in range(10):
        print('class {0} accuracy: {1}'.format(i, class_accuracy[i]))

    # eval model's accuacy in cw adversarial examples
    cw_accuracy = model_eval(sess, x, y, pred, cw_adv, adv_ys,
                             feed=eval_feed,
                             args={'batch_size': 128})
    print('model cw_accuracy: {0}'.format(cw_accuracy))

    part_X = X[:cw_adv.shape[0]]
    #eval adv's l2 distance
    l2_dis = calculate_l2_dis(part_X/255, cw_adv/255)
    print('adversarial examples\' mean l2 distance: {0}'.format(l2_dis))

    # show and save img
    import numpy as np
    adv_imgs = np.around(cw_adv).astype(int)
    print(np.max(adv_imgs))
    compare_show(X[16], adv_imgs[16])
    import matplotlib
    matplotlib.image.imsave('cw.png', adv_imgs[16])

    # eval model's uncertainty
    dropout_num = 30
    uncert = evaluate_uncertainty(sess, model, x, part_X, dropout_num,
                                  batch_size, nb_classes, train_feed)

    # eval model's cw_uncertainty
    cw_uncert = evaluate_uncertainty(sess, model, x, cw_adv, dropout_num,
                                     batch_size,nb_classes, train_feed)

    # plot uncertainty histogram
    plt.figure("uncertainty_X")
    n, bins, patches = plt.hist(uncert, bins=25,edgecolor='None',facecolor='blue')
    plt.show()

    plt.figure('uncertainty_CW')
    cw_n, cw_bins, cw_patches = plt.hist(cw_uncert, bins=25,
                                         edgecolor='None',facecolor='red')
    plt.show()

    plt.figure('uncertainty_collections')
    plt.hist(uncert, bins=25,edgecolor='None',facecolor='blue')
    plt.hist(cw_uncert, bins=25,edgecolor='None',facecolor='red')
    plt.show()

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
    # model = make_cnn_drop_model(name = 'cnn_0.9_dropout_0.5_model')
    # model = make_vgg16_model(name = 'vgg16_model')
    # model = make_vgg16_clipRelu_model(name='vgg16_clipReLu', eval_mode=True)
    # model = make_vgg16_clipRelu_ordering_exchange_model(name='vgg16_clipRelu_ordering_exchange')
    mdt(model, FLAGS.data_dir, FLAGS.checkpoint_dir,
        FLAGS.train_dir, FLAGS.adversarial_dir, FLAGS.batch_size)

if __name__ == '__main__':
    tf.app.run()
