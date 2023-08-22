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
from tensorflow.keras.layers import Input
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
import dataset_utils

# Suppress tensorflow INFO and WARNING messages
# Source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ContactClassifierNetwork:
	def __init__(self, model_path=None, color_mode='rgb', preprocess_method='EfficientNet',
				 img_target_size=(456, 456), batch_size=16, positive_indx=0, crop_center=False):
		"""

		:param model_path: Path to keras model
		:param color_mode: 'grayscale', 'rgb', 'rgba', default=grayscale. Used with tensorflow.keras load_img method
		:param preprocess_method: Must be one of ['ResNet', 'EfficientNet'] or None. If none, divide the image by 255 for inference. 
		:param img_target_size: Size images will be rescaled to before feeding through network
		:param batch_size: Split images into groups of batch_size to feed into the network at once.
							The last batch processed may be smaller than batch_size
		:param positive_indx: Index in model prediction vector corresponding to "positive"
		:param crop_center: If true, crop center of the image to img_target_size to retain aspect ration. Otherwise resize with interpolation
		"""
		assert model_path is not None, "Please provide a path to the contact classifier network model"

		if preprocess_method is not None:
			assert preprocess_method in ['ResNet', 'EfficientNet']

		self.model = None
		self.color_mode = color_mode
		self.preprocess_function = None
		if preprocess_method == 'ResNet':
			from tensorflow.keras.applications.resnet50 import preprocess_input
			self.preprocess_function = preprocess_input
		elif preprocess_method == 'EfficientNet':
			from tensorflow.keras.applications.efficientnet import preprocess_input
			self.preprocess_function = preprocess_input
		self.img_target_size = img_target_size
		self.batch_size = batch_size
		self.positive_indx = positive_indx
		self.crop_center = crop_center
		self.logger = logging.getLogger('contactclassifier.ContactClassifierNetwork')

		num_channels = 1
		if self.color_mode == 'rgb':
			num_channels = 3
		if self.color_mode == 'rgba':
			num_channels = 4
		self.img_target_size = (self.img_target_size[0], self.img_target_size[1], num_channels)

		self.logger.info('Loading model...')
		model_no_preprocess = load_model(model_path)
		self.model = self.add_model_preprocessing(model_no_preprocess)
		del model_no_preprocess
		# self.model._make_predict_function()
		self.logger.info('Model loaded...')

	def add_model_preprocessing(self, model):
		input = Input(shape=self.img_target_size)
		if self.preprocess_function is not None:
			model_input = self.preprocess_function(input)
		else:
			model_input = input/255.0
		model_output = model(model_input)
		return Model(inputs=input, outputs=model_output)

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
		return np.array(load_img_tf(img_path, color_mode=self.color_mode, target_size=self.img_target_size))

	def load_batch(self, img_paths):
		num_imgs = len(img_paths)
		img_batch = np.zeros((num_imgs, self.img_target_size[0], self.img_target_size[1], self.img_target_size[2]))
		for i in range(num_imgs):
			if self.crop_center:
				img = np.array(dataset_utils.load_img_center_crop(img_paths[i], color_mode=self.color_mode, crop_size=self.img_target_size[:2]))
			else:
				img = np.array(self.load_img(img_paths[i]))
				if self.color_mode == 'grayscale':
					img = np.expand_dims(img, -1)
			img_batch[i] = img
		return img_batch

	def processFiles(self, list_images):
		all_preds = []
		num_process_imgs = len(list_images)
		for i in range(0, num_process_imgs, self.batch_size):
			list_batch_images = list_images[i:i+self.batch_size]
			img_batch = self.load_batch(list_batch_images)

			preds = self.model.predict(img_batch)

			# Use only positive prediction score to make decision
			preds = preds[:, self.positive_indx]
			all_preds += list(preds)
			
		return all_preds
	
	def processFile(self, img_path):
		return self.processFiles([img_path])

