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
from tensorflow.keras.applications.resnet50 import preprocess_input
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

class ContactClassifierNetwork:
	def __init__(self, model_path=None, color_mode='grayscale',  resnet_preprocess=False, img_target_size=(640, 640),
				 batch_size=16, positive_indx=0):
		"""

		:param model_path: Path to keras model
		:param color_mode: 'grayscale', 'rgb', 'rgba', default=grayscale. Used with tensorflow.keras load_img method
		:param resnet_preprocess: Whether to apply resnet preprocessing to image before inference. If false, divide
									image by 255 before inference
		:param img_target_size: Size images will be rescaled to before feeding through network
		:param batch_size: Split images into groups of batch_size to feed into the network at once.
							The last batch processed may be smaller than batch_size
		:param positive_indx: Index in model prediction vector corresponding to "positive"
		"""
		assert model_path is not None, "Please provide a path to the contact classifier network model"

		self.model = None
		self.color_mode = color_mode
		self.resnet_preprocess = resnet_preprocess
		self.img_target_size = img_target_size
		self.batch_size = batch_size
		self.positive_indx = positive_indx
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
		if self.resnet_preprocess:
			model_input = preprocess_input(input)
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
			img = self.load_img(img_paths[i])
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

