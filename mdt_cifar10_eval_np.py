import numpy as np
import os
import tensorflow as tf

import mdt_cifar10_input
import cifar10_data
import MDT_model
from cleverhans.utils_tf import tf_model_load
import matplotlib.pyplot as plt  

DATA_DIR = './tmp/cifar10_data'
CHEKPOINT_DIR = './tmp/cifar10_train'


def checkpoint_load(sess, checkpoint_dir, moving_variables=None):
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		if moving_variables:
			variable_averages = tf.train.ExponentialMovingAverage(0.9)
			variables_to_restore = variable_averages.variables_to_restore(moving_variables)
			saver = tf.train.Saver(variables_to_restore)
			saver.restore(sess, ckpt.model_checkpoint_path)
			print(sess.run(moving_variables))
		tf_model_load(sess, ckpt.model_checkpoint_path)
		return True

	print('restore fails: please provide correct checkpoint directory')
	return False

def evaluate_uncertainty(sess, model, x, X, dropout_num,
						 batch_size, nb_classes, feed=None):
	pred = model(x)
	md_pred = []

	def predict():
		n_batches = int(np.ceil(X.shape[0]/ float(batch_size)))
		output = np.zeros(shape=(len(X), nb_classes))
		for i in range(n_batches):
			feed_dict = {x: X[i*batch_size:(i+1)*batch_size]}
			if feed is not None:
				feed_dict.update(feed)
			output[i*batch_size:(i+1)*batch_size] = \
				sess.run(pred, feed_dict=feed_dict)
		return output
	
	for i in range(dropout_num):
		md_pred.append(predict())

	md_pred = np.asarray(md_pred)
	uncert = md_pred.var(axis=0).mean(axis=1)

	return uncert

def calculate_l2_dis(X, adv_X):
	dif = X - adv_X
	dif = np.reshape(dif, (dif.shape[0],-1))
	sqrt = np.linalg.norm(dif, axis=1)
	mean = np.mean(sqrt)
	return mean

def compare_show(ori, adv):
	plt.figure()
	plt.subplot(2,2,1)
	plt.imshow(ori)
	plt.subplot(2,2,2)
	plt.imshow(adv)
	plt.show()

def main(argv=None):
	sess = tf.Session()
	model = MDT_model.make_mdt_model()
	evaluate_np(sess, model, CHEKPOINT_DIR, DATA_DIR)

if __name__ =='__main__':
	tf.app.run()