from PIL import Image
import numpy as np
import tensorflow as tf

def crop_center(img, crop_size):
	h, w, c = img.shape
	crop_h, crop_w = crop_size	
	h_offset = max((h - crop_h) // 2, 0)
	w_offset = max((w - crop_w) // 2, 0)
	cropped_img = img[h_offset:h-h_offset, w_offset:w-w_offset, :]
	return cropped_img

def resize_img(img, target_size, color_mode='grayscale'):
	if color_mode == 'grayscale':
		img = np.dstack((img, img, img))
	resized_img = np.array(Image.fromarray(img).resize(target_size, resample=Image.NEAREST))
	if color_mode == 'grayscale':
		resized_img = np.expand_dims(resized_img[:, :, 0], -1)
	return resized_img

def load_img_center_crop(img_path, crop_size, color_mode='grayscale'):
	img = tf.keras.preprocessing.image.load_img(img_path, color_mode=color_mode)
	img = tf.keras.preprocessing.image.img_to_array(img)
	# Images with dimension(s) less than specified by crop_size will be interpolated
	cropped_img = crop_center(img, crop_size)
	cropped_img = np.array(cropped_img, dtype=np.uint8)
	cropped_img = resize_img(cropped_img, crop_size, color_mode)
	return cropped_img

def load_imgs_center_crop(img_path_l, crop_size, color_mode='grayscale'):
	'''Returns loaded images in same order as specified in the provided list'''
	loaded_imgs = []
	for img_path in img_path_l:
		loaded_imgs.append(load_img_center_crop(img_path, crop_size, color_mode))
	return np.array(loaded_imgs)

def get_categorical_labels(df, label_field):
	'''
	Converts labels into one-hot vectors. Labels are strings. 
	Returned unique_labels is ordered in correspondence with the one-hot-vector-labels (e.g., if there were 3 classes,
	unique_labels[0] corresponds to [1, 0, 0], unique_labels[1] corresponds to [0, 1, 0], etc.)
	'''
	labels = list(df[label_field])
	unique_labels = list(set(labels))
	unique_labels.sort()
	num_classes = len(unique_labels)
	numerical_labels = []
	for l in labels:
		for i, ul in enumerate(unique_labels):
			if ul == l:
				numerical_labels.append(i)
	return tf.keras.utils.to_categorical(numerical_labels, num_classes=num_classes), unique_labels
