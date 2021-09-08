'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Lambda
import numpy as np
import os
import inspect
from .features.extract_features import azimuthal_average_2d_tf_py_func, bsif_tf_py_func, load_bsif_filter


base_path = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
bsif_filter_path = os.path.join(base_path, 'features/models/bsif_filters/ICAtextureFilters_7x7_9bit.mat')

texturefilters = load_bsif_filter(bsif_filter_path)
texturefilters = np.array(texturefilters, dtype=np.float32)


def bsif_features(x, mode='nh'):
    assert mode in ['im', 'h', 'nh'], "Give mode \'" + mode + "\' is not available."

    # Remove alpha channel if rgba
    if x.get_shape()[-1] == 4:
        x = x[:, :, :3]
    # Convert RGB to grayscale
    if x.get_shape()[-1] == 3:
        x = tf.image.rgb_to_grayscale(x)

    numScl = texturefilters.shape[2]
    codeImg = tf.ones(tf.shape(x))

    # Remove the last dimension (should be 1 at this point)
    x = tf.squeeze(x, axis=-1)

    r = int(np.floor(texturefilters.shape[0] / 2))

    upimg = x[:,:r,:]
    btimg = x[:,-r:, :]

    lfimg = x[:,:,:r]
    rtimg = x[:,:,-r:]

    cr11 = x[:,:r,:r]
    cr12 = x[:,:r,-r:]
    cr21 = x[:,-r:,:r]
    cr22 = x[:,-r:,-r:]

    imgWrap = tf.concat([
        tf.concat([cr22, btimg, cr21], axis=2),
        tf.concat([rtimg, x, lfimg], axis=2),
        tf.concat([cr12, upimg, cr11], axis=2),
    ], axis=1)
    imgWrap = tf.expand_dims(imgWrap, -1)

    strides = [1, 1, 1, 1]
    padding = "VALID"
    for i in range(1, numScl + 1):
        tmp = texturefilters[:, :, numScl - i]
        tmp = tf.expand_dims(tf.expand_dims(tmp, -1), -1)
        ci = tf.nn.conv2d(imgWrap, tmp, strides, padding)
        ci_gt_zero = ci > 0
        ci_gt_zero = tf.cast(ci_gt_zero, tf.float32)
        codeImg += ci_gt_zero * (2 ** (i - 1))

    if mode == 'im':
        bsifdescription = codeImg
    if mode == 'h' or mode == 'nh':
        bsifdescription = tfp.stats.histogram(codeImg, list(range(1, 2 ** numScl + 1)), axis=[1, 2, 3])
        bsifdescription = tf.transpose(bsifdescription)
    if mode == 'nh':
        bsifdescription = bsifdescription / tf.expand_dims(tf.reduce_sum(bsifdescription, axis=1), -1)

    return bsifdescription

def bsif_features_np_func(x):
    feature_size = 511
    bsif_features = tf.numpy_function(bsif_tf_py_func, [x, feature_size], tf.float32)
    bsif_features.set_shape((None, feature_size))
    return bsif_features

def fft_features_np_func(x):
    interp_num = 300
    # Remove alpha channel if rgba
    if x.get_shape()[-1] == 4:
        x = x[:, :, :3]
    # Convert RGB to grayscale
    if x.get_shape()[-1] == 3:
        x = tf.image.rgb_to_grayscale(x)
    x = tf.cast(x, tf.complex64)
    fft_img = tf.signal.fft2d(x)
    fft_img = tf.signal.fftshift(fft_img)
    magnitude_spectrum = 20*tf.math.log(tf.math.abs(fft_img) + 1e-8)
    averaged_fft_features = tf.numpy_function(azimuthal_average_2d_tf_py_func, [magnitude_spectrum, interp_num], tf.float32)
    averaged_fft_features.set_shape((None, interp_num))
    return averaged_fft_features

def fft_mag_and_phase_img(x):
    x = tf.cast(x, tf.complex64)
    fft_img = tf.signal.fft2d(x)
    fft_img = tf.signal.fftshift(fft_img)
    magnitude = tf.math.abs(fft_img)
    phase = tf.math.angle(fft_img)
    return tf.keras.layers.concatenate([magnitude, phase])

def FFTAzimuthalAverage():
    return Lambda(fft_features_np_func, output_shape=(300,), mask=None, arguments=None, name="fft_azimuthal_avg")

def FFTMagAndPhaseImg():
    return Lambda(fft_mag_and_phase_img, output_shape=None, mask=None, arguments=None, name="fft_img")

def BSIFFeatures():
    return Lambda(bsif_features, output_shape=(511,), mask=None, arguments=None, name="bsif_features")

def NumpyBSIFFeatures():
    return Lambda(bsif_features_np_func, output_shape=(511,), mask=None, arguments=None, name="bsif_features")
