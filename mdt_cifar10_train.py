import tensorflow as tf
import time
from datetime import datetime
import os

import mdt_cifar10_input
import MDT_model
import cifar10_data as data
from mdt_cifar10_eval_np import checkpoint_load
import math
# import cleverhans
# from cleverhans.utils_tf import model_train, model_eval, batch_eval

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('train_dir', './tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = mdt_cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mdt_cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = mdt_cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 0.5      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.95  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1      # Initial learning rate.
MAX_EPOCH = 500
MAX_STEPS = math.ceil(MAX_EPOCH*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size)
#LOG_FREQUENCY = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size
LOG_FREQUENCY = 10


def stand_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def adv_net_loss(input, model, labels, target, adv_output_layer, confidence, c):
    # calculate l2 distance between ori_input and adversarial examples
    adv_output = model.get_layer(input, adv_output_layer)
    dif = tf.subtract(adv_output, input)
    # reshape_dif = tf.reshape(dif, shape=(dif.get_shape()[0],-1))
    # l2_dis_loss = tf.norm(reshape_dif, axis=1)
    l2_dis_loss = tf.square(dif)
    l2_dis_loss = tf.reduce_mean(l2_dis_loss, name='l2_dis_loss')

    tf.add_to_collection('losses', l2_dis_loss)

    # attack target loss
    logits = model(input)
    one_hot_labels = tf.one_hot(labels,10)
    real = tf.reduce_sum(one_hot_labels*logits, 1)
    other_max = tf.reduce_max((1-one_hot_labels)*logits-one_hot_labels*10000, 1)
    if target:
        attack_loss = tf.maximum(0.0, other_max - real + confidence)
    else:
        attack_loss = tf.maximum(0.0, real - other_max + confidence)
    attack_loss = tf.reduce_mean(attack_loss, name='attack_loss')

    tf.add_to_collection('losses', attack_loss)

    # total loss
    total_loss = l2_dis_loss*c + attack_loss*0

    return total_loss

def adv_loss(dis_loss, logits, labels, target, confidence, c):

    tf.add_to_collection('adv_losses', dis_loss)

    # attack target loss
    one_hot_labels = tf.one_hot(labels,10)
    if not target:
        adv_labels = tf.argmax((1-one_hot_labels)*logits-one_hot_labels*10000, 1)
    else:
        adv_labels = labels

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=adv_labels, logits=logits, name='cross_entropy_per_example'
    )
    attack_loss = tf.reduce_mean(cross_entropy, name='attack_loss')

    tf.add_to_collection('adv_losses', attack_loss)

    # discrimiator_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=labels, logits=logits, name='cross_entropy_per_example'
    # )
    
    # add ratio to attack loss
    # ratio = -tf.log(dis_loss)
    # ratio = tf.stop_gradient(ratio)

    # total loss
    total_loss = tf.add(attack_loss,-tf.log(1-dis_loss)*c,'adv_total_loss')
    # total_loss = tf.subtract(ratio*attack_loss,tf.log(1-dis_loss)*c,'total_loss')
    # total_loss = tf.cond(dis_loss>0.01, lambda:ratio*attack_loss*1-tf.log(1-dis_loss)*c, lambda: ratio*attack_loss, 'total_loss')

    return total_loss

def loss_summary(total_loss, name=None):
    if name:
        losses = tf.get_collection(name)
    else:
        losses = tf.get_collection('losses')
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)


def create_train_op(total_loss, global_step, var_list=None, opt=None):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    if opt is None:
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)
        # Compute gradients.
        opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss, var_list)

    loss_summary(total_loss, 'adv_losses')
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if var_list is None:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
    else:
        for var in var_list:
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    if var_list is None:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    else:
        update_ops=[]
    update_ops.append(apply_gradient_op)
    # print(update_ops)
    with tf.control_dependencies(update_ops):
        train_op = tf.no_op(name='train')

    return train_op

def create_alter_train_op(adv_loss, det_loss, adv_vars, det_vars, adv_opt, det_opt):
    
    adv_grads = adv_opt.compute_gradients(adv_loss, adv_vars)
    det_grads = det_opt.compute_gradients(det_loss, det_vars)

    loss_summary(adv_loss, name='adv_losses')
    loss_summary(det_loss, name='losses')

    # Apply gradients.
    global_step = tf.train.get_or_create_global_step()
    apply_adv_gradient_op = adv_opt.apply_gradients(adv_grads, global_step=global_step)
    apply_det_gradient_op = det_opt.apply_gradients(det_grads, global_step=global_step)


    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in adv_grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/adv_gradients', grad)

    # Add histograms for gradients.
    for grad, var in det_grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/det_gradients', grad)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    adv_update_ops = [apply_adv_gradient_op]
    det_update_ops = [apply_det_gradient_op] + update_ops

    # print('det_update_ops.....',update_ops)

    with tf.control_dependencies(adv_update_ops):
        adv_train_op = tf.no_op(name='adv_train')

    with tf.control_dependencies(det_update_ops):
        det_train_op = tf.no_op(name='det_train')


    return adv_train_op, det_train_op

def train_process(model, loss, images, labels, train_dir='./tmp/cifar10_train',
                  batch_size=128, var_list=None,checkpoint_dir=None,
                  feed=None,
                  log_device_placement=False):


    logits = model(images)

    # calculate loss
    # total_loss = loss(logits, labels)

    global_step = tf.train.get_or_create_global_step()
    train_op = create_train_op(loss, global_step, var_list)

    # calculate training accuracy
    correct_pre = tf.equal(tf.argmax(logits, 1), labels)
    train_acc = tf.reduce_mean(
        tf.cast(correct_pre, 'float'), name='train_accuracy')
    tf.summary.scalar(train_acc.op.name, train_acc)

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            if self._step % LOG_FREQUENCY == 0:
                return tf.train.SessionRunArgs([loss, train_acc])

        def after_run(self, run_context, run_values):
            if self._step % LOG_FREQUENCY == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value, train_acc = run_values.results
                examples_per_sec = LOG_FREQUENCY * batch_size / duration
                sec_per_batch = float(duration / LOG_FREQUENCY)

                format_str = ('%s: step %d, loss = %.2f, train_acc = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), self._step, loss_value,
                                    train_acc, examples_per_sec, sec_per_batch))

    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        scaffold=scaffold,
        config=tf.ConfigProto(
            log_device_placement=log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():
            if feed is not None:
                mon_sess.run(train_op, feed_dict=feed)
            else:
                mon_sess.run(train_op)

def train_adv_net(sess, model, loss, images, labels, train_dir='./tmp/cifar10_train',
                  batch_size=128, var_list=None,checkpoint_dir=None,
                  feed=None,
                  log_device_placement=False):
    logits = model(images)

    # calculate loss
    # total_loss = loss(logits, labels)

    global_step = tf.train.get_or_create_global_step()
    train_op = create_train_op(loss, global_step, var_list)

    # calculate training accuracy
    correct_pre = tf.equal(tf.argmax(logits, 1), labels)
    train_acc = tf.reduce_mean(
        tf.cast(correct_pre, 'float'), name='train_accuracy')
    tf.summary.scalar(train_acc.op.name, train_acc) 

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

    # g_list = tf.global_variables()
    # bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    # print(sess.run(bn_moving_vars))

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      step = 0
      while step < MAX_STEPS and not coord.should_stop():
        _, l, acc = sess.run([train_op, loss, train_acc], feed_dict=feed)
        # l, acc = sess.run([loss, train_acc], feed_dict=feed)
        print('{0} step, loss: {1} train_acc:{2}'.format(step, l, acc))
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def train_adv_encoder(sess, logits, loss, labels, train_op, train_dir='./tmp/cifar10_train',
                  batch_size=128, feed=None, nb_epochs=500):

    global_step = tf.train.get_or_create_global_step()

    # calculate training accuracy
    correct_pre = tf.equal(tf.argmax(logits, 1), labels)
    train_acc = tf.reduce_mean(
        tf.cast(correct_pre, 'float'), name='adv_train_accuracy')
    tf.summary.scalar(train_acc.op.name, train_acc) 

    # summary writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_dir + 'summary',
                                         tf.get_default_graph())

    # calculate max steps
    max_steps = math.ceil(nb_epochs*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/\
                          batch_size)

    # create model saver
    to_save_variables = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    to_save_variables += bn_moving_vars+[global_step]

    saver = tf.train.Saver(to_save_variables)

    # create threads to fetch input data
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      step = 1
      while step < max_steps and not coord.should_stop():
        # l, acc = sess.run([loss, train_acc], feed_dict=feed)
        if step%10==0:
            _, l, acc = sess.run([train_op, loss, train_acc], feed_dict=feed)
            print('{0} step, loss: {1} train_acc:{2}'.format(step, l, acc))
        else:
            sess.run([train_op], feed_dict=feed)

        # # write summary
        # summary = sess.run(merged)
        # train_writer.add_summary(summary, step)

        if step%100==0:
            # write summary
            summary = sess.run(merged, feed_dict=feed)
            train_writer.add_summary(summary, step)

        if step%1000==0:
            # # write summary
            # summary = sess.run(merged)
            # train_writer.add_summary(summary, step)
            saver.save(sess, train_dir, global_step)
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def adversarial_train(sess, logits, loss, labels, adv_train_op, det_train_op, train_dir='./tmp/cifar10_train',
                  batch_size=128, feed=None, nb_epochs=500):

    global_step = tf.train.get_or_create_global_step()

    # calculate training accuracy
    correct_pre = tf.equal(tf.argmax(logits, 1), labels)
    train_acc = tf.reduce_mean(
        tf.cast(correct_pre, 'float'), name='adv_train_accuracy')
    tf.summary.scalar(train_acc.op.name, train_acc) 

    # summary writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_dir + 'summary',
                                         tf.get_default_graph())

    # calculate max steps
    max_steps = math.ceil(nb_epochs*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/\
                          batch_size)

    # create model saver
    to_save_variables = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    to_save_variables += bn_moving_vars+[global_step]

    saver = tf.train.Saver(to_save_variables)

    # create threads to fetch input data
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      step = 1
      while step < max_steps and not coord.should_stop():
        # l, acc = sess.run([loss, train_acc], feed_dict=feed)
        if step%10==0:
            _, _, l, acc = sess.run([adv_train_op, det_train_op, loss, train_acc], feed_dict=feed)
            print('{0} step, loss: {1} train_acc:{2}'.format(step, l, acc))
        else:
            sess.run([adv_train_op, det_train_op], feed_dict=feed)

        # # write summary
        # summary = sess.run(merged)
        # train_writer.add_summary(summary, step)

        if step%100==0:
            # write summary
            summary = sess.run(merged, feed_dict=feed)
            train_writer.add_summary(summary, step)

        if step%1000==0:
            # # write summary
            # summary = sess.run(merged)
            # train_writer.add_summary(summary, step)
            saver.save(sess, train_dir, global_step)
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):
    # download data if not exits
    data.maybe_download_and_extract_binary(FLAGS.data_dir)

    # define model
    model = MDT_model.make_mdt_model()

    # train model
    train_process(model, stand_loss, FLAGS.data_dir, FLAGS.train_dir, FLAGS.batch_size, FLAGS.log_device_placement)


if __name__ == '__main__':
    tf.app.run()
