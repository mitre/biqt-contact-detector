'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array 
from tensorflow.keras.preprocessing.image import load_img as load_img_tf
from tensorflow.keras.layers import Input, Concatenate, Lambda
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import logging

import sys
import os
import os.path as osp
sys.path.append(os.environ['BIQT_HOME'] + '/providers/BIQTContactDetector/src/python/inference')

from PIL import Image

# if using lambda in cv_features
# /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers/core.py:986: UserWarning: cv_featur
# es.tf_features is not loaded, but a Lambda layer uses it. It may cause errors.
from cv_features.tf_features import bsif_features
from cv_features.tf_features import fft_features_np_func

# Suppress tensorflow INFO and WARNING messages
# Source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_to_rgb_np(imgs, target_size):
	converted_imgs = np.zeros((imgs.shape[0], target_size[0], target_size[1], 3))
	for i, img in enumerate(imgs):
		img = np.array(img, dtype=np.uint8)
		converted_imgs[i] = np.array(Image.fromarray(img).convert('RGB').resize(target_size, Image.NEAREST))
	return np.array(converted_imgs, dtype=np.float32)
def convert_to_rgb(imgs, target_size):
	return tf.numpy_function(convert_to_rgb_np, [imgs, target_size], tf.float32)
def ToRGB(target_size):
	return Lambda(lambda imgs: convert_to_rgb(imgs, target_size), output_shape=target_size)

def convert_to_grayscale_np(imgs, target_size):
	converted_imgs = np.zeros((imgs.shape[0], target_size[0], target_size[1]))
	for i, img in enumerate(imgs):
		img = np.array(img, dtype=np.uint8)
		converted_imgs[i] = np.array(Image.fromarray(img).convert('L').resize(target_size, Image.NEAREST))
	return np.expand_dims(np.array(converted_imgs, dtype=np.float32), -1)
def convert_to_grayscale(imgs, target_size):
	return tf.numpy_function(convert_to_grayscale_np, [imgs, target_size], tf.float32)
def ToGrayScale(target_size):
	return Lambda(lambda imgs: convert_to_grayscale(imgs, target_size), output_shape=target_size)


class DualContactClassifierNetwork:
	def __init__(self, cosmetic_model_path=None, soft_lens_model_path=None, cosmetic_color_mode='grayscale', soft_lens_color_mode='rgb',
				 resnet_preprocess_cosmetic=False, resnet_preprocess_soft_lens=True, img_target_size=(640, 640),
				 batch_size=16, cosmetic_positive_indx=0, soft_lens_positive_indx=1):
		"""
		Uses two models to perform prediction of whether a given iris image contains a cosmetic contact lens, a clear/soft
		contact lens, or no contact lens. 
		Returns two sets of confidence: 
			1. Confidence that the image contains a cosmetic contact lens
			2. Confidence that the image contains a clear/soft-lens

		If confidence for cosmetic contact lens is low, the second soft-lens prediction can be used 
		to distinguish between soft-lens and non-contact-lens images.

		:param cosmetic_model_path: Path to cosmetic vs non-cosmetic contact lens keras model
		:param soft_lens_model_path: Path to soft vs non-lens keras model
		:param cosmetic_color_mode: 'grayscale', 'rgb'
		:param soft_lens_color_mode: 'grayscale', 'rgb'
		:param resnet_preprocess_cosmetic: Whether to apply resnet preprocessing to image before inference for cosmetic model.
											If false, divide image by 255 before inference
		:param resnet_preprocess_soft_lens: Whether to apply resnet preprocessing to image before inference for soft lens model
											If false, divide image by 255 before inference
		:param img_target_size: Size images will be rescaled to before feeding through network.
								Both cosmetic and soft lens should be able to input images this size
		:param batch_size: Split images into groups of batch_size to feed into the network at once.
							The last batch processed may be smaller than batch_size
		:param cosmetic_positive_indx: Index in cosmetic model prediction vector corresponding to "positive"
		:param soft_lens_positive_indx: Index in soft lens model prediction vector corresponding to "positive"
		"""
		assert cosmetic_model_path is not None, "Please provide a path to the cosmetic lens model"
		assert soft_lens_model_path is not None, "Please provide a path to the soft lens model"

		self.model = None
		self.cosmetic_color_mode = cosmetic_color_mode
		self.soft_lens_color_mode = soft_lens_color_mode
		self.resnet_preprocess_cosmetic = resnet_preprocess_cosmetic
		self.resnet_preprocess_soft_lens = resnet_preprocess_soft_lens
		self.img_target_size = (img_target_size[0], img_target_size[1], 3)
		self.batch_size = batch_size
		self.cosmetic_positive_indx = cosmetic_positive_indx
		self.soft_lens_positive_indx = soft_lens_positive_indx
		self.logger = logging.getLogger('contactclassifier.DualContactClassifier')

		self.logger.info('Loading model...')
		cosmetic_model = load_model(cosmetic_model_path)
		soft_lens_model = load_model(soft_lens_model_path)
		self.cosmetic_num_preds = cosmetic_model.layers[-1].output_shape[1]
		self.soft_lens_num_preds = soft_lens_model.layers[-1].output_shape[1]
		self.model = self.merge_models(cosmetic_model, soft_lens_model)
		del cosmetic_model
		del soft_lens_model
		# self.model._make_predict_function()
		self.logger.info('Model loaded...')

	def merge_models(self, cosmetic_model, soft_lens_model):
		# Prevents layer names from clashing
		soft_lens_model._name = soft_lens_model.name + '_2'
		for layer in soft_lens_model.layers:
			layer._name = layer.name + str('_2')

		input = Input(shape=(None, None, None))
		rgb_input = ToRGB((self.img_target_size[0], self.img_target_size[1]))(input)
		gray_input = ToGrayScale((self.img_target_size[0], self.img_target_size[1]))(input)

		# Choose color mode for cosmetic lens network
		if self.cosmetic_color_mode == 'grayscale':
			cosmetic_input = gray_input
		else:
			cosmetic_input = rgb_input
		# Choose preprocessing for cosmetic lens network
		if self.resnet_preprocess_cosmetic:
			cosmetic_input = preprocess_input(cosmetic_input)
		else:
			cosmetic_input /= 255.0
		cosmetic_output = cosmetic_model(cosmetic_input)

		# Choose color mode for soft lens network
		if self.soft_lens_color_mode == 'grayscale':
			soft_lens_input = gray_input
		else:
			soft_lens_input = rgb_input
		# Choose preprocessing for soft lens network
		if self.resnet_preprocess_soft_lens:
			soft_lens_input = preprocess_input(soft_lens_input)
		else:
			soft_lens_input /= 255.0
		soft_lens_output = soft_lens_model(soft_lens_input)

		combined_output = Concatenate()([cosmetic_output, soft_lens_output])
		return Model(inputs=input, outputs=combined_output)

	def is_img_path(self, path):
		try:
			Image.open(path)
			return True
		except:
			return False

	def load_img(self, img_path):
		if not osp.isfile(img_path):
			raise FileNotFoundError('Image ' + img_path + ' could not be found')
		if not self.is_img_path(img_path):
			raise IOError('File ' + img_path + ' could not be read as an image')
		return load_img_tf(img_path)

	def load_batch(self, img_paths):
		num_imgs = len(img_paths)
		img_batch = []
		for i in range(num_imgs):
			img = np.array(self.load_img(img_paths[i]))
			# Remove alpha channel
			if len(img.shape) == 4:
				img = img[:, :, :3]
			img_batch.append(img)
		return np.array(img_batch)

	def processFiles(self, list_images):
		all_cosmetic_preds = []
		all_soft_lens_preds = []
		num_process_imgs = len(list_images)
		for i in range(0, num_process_imgs, self.batch_size):
			list_batch_images = list_images[i:i+self.batch_size]
			img_batch = self.load_batch(list_batch_images)

			combined_preds = self.model.predict(img_batch)

			# Split out cosmetic and soft lens predictions
			cosmetic_preds = combined_preds[:, :self.cosmetic_num_preds]
			soft_lens_preds = combined_preds[:, self.cosmetic_num_preds:]

			# Only use predictions at positive-class index
			cosmetic_preds = cosmetic_preds[:, self.cosmetic_positive_indx]
			soft_lens_preds = soft_lens_preds[:, self.soft_lens_positive_indx]

			all_cosmetic_preds += list(cosmetic_preds)
			all_soft_lens_preds += list(soft_lens_preds)

		return all_cosmetic_preds, all_soft_lens_preds

	def processFile(self, img_path):
		cosmetic_preds, soft_lens_preds =  self.processFiles([img_path])
		return [cosmetic_preds[0], soft_lens_preds[0]]
