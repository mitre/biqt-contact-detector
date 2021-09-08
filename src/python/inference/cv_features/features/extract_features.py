# -*- coding: utf-8 -*-
import os
import inspect

import numpy as np
from PIL import Image
from scipy.interpolate import griddata
from skimage.color import rgb2gray
from scipy.signal import convolve2d, correlate2d

from scipy.ndimage import gaussian_laplace

from . import radialProfile
from .convert_mat_struct import loadmat

base_path = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
bsif_filter_path = os.path.join(base_path, 'models/bsif_filters/ICAtextureFilters_7x7_9bit.mat')


class Feature:
    BSIF = 1
    FFT_AZIMUTHAL_AVERAGE = 2
    LOG_MEAN_POWER = 3

    str_to_enum = {
        'bsif': BSIF,
        'fft' : FFT_AZIMUTHAL_AVERAGE,
        'LoG' : LOG_MEAN_POWER
    }

def load_bsif_filter(filter_file):
    texturefilters = loadmat(filter_file)['ICAtextureFilters']
    return texturefilters


texturefilters = load_bsif_filter(bsif_filter_path)

def bsif_tf_py_func(imgs, feature_size):
    bsif_results = np.zeros((imgs.shape[0], feature_size))
    for i in range(imgs.shape[0]):
        bsif_results[i] = bsif(imgs[i])
    return np.array(bsif_results, np.float32)

def bsif(img, mode='nh'):
    """
    Extracts bsif features. Port of MATLAB code from http://www.ee.oulu.fi/~jkannala/bsif/bsif.html

    :param img: RGB image in the form of numpy array
    :param texturefilters: BSIF filter
    :param mode: 'im' for binary image 'h' for histogram of binary image, 'nh' for normalizes histogram of binary image
    :return: bsif features
    """
    assert mode in ['im', 'h', 'nh'], "Give mode \'" + mode + "\' is not available."

    img = img*255
    img[img>255] = 255
    img[img<0] = 0
    img = np.array(img, np.uint8)
    if img.shape[-1] == 1:
        img = np.dstack((img, img, img))

    img = np.array(Image.fromarray(img).convert('LA'))[:,:,0]

    numScl = texturefilters.shape[2]
    codeImg = np.ones(img.shape)

    r = int(np.floor(texturefilters.shape[0]/2))

    upimg = img[:r,:]
    btimg = img[-r:, :]

    lfimg = img[:,:r]
    rtimg = img[:,-r:]

    cr11 = img[:r,:r]
    cr12 = img[:r,-r:]
    cr21 = img[-r:,:r]
    cr22 = img[-r:,-r:]

    imgWrap = np.vstack((
                np.hstack((cr22, btimg, cr21)),
                np.hstack((rtimg, img, lfimg)),
                np.hstack((cr12, upimg, cr11))))

    for i in range(1, numScl+1):
        tmp = texturefilters[:,:,numScl-i]
        ci = correlate2d(imgWrap, tmp, mode='valid')
        codeImg += (ci > 0) * (2 ** (i-1))

    if mode == 'im':
        bsifdescription = codeImg
    if mode == 'h' or mode == 'nh':
        bsifdescription = np.histogram(codeImg, list(range(1,2**numScl + 1)))[0]
    if mode == 'nh':
        bsifdescription = bsifdescription / np.sum(bsifdescription)

    return bsifdescription

def azimuthal_average_2d_tf_py_func(spectrum_imgs, interp_num):
    azimuthal_average_results = np.zeros((spectrum_imgs.shape[0], interp_num))
    for i in range(spectrum_imgs.shape[0]):
        azimuthal_average_results[i] = azimuthal_average_2d(np.squeeze(spectrum_imgs[i]), interp_num=interp_num)
    return np.array(azimuthal_average_results, np.float32)

def azimuthal_average_2d(spectrum_img, interp_num = 300):
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(spectrum_img)

    # Interpolation
    points = np.linspace(0, interp_num, num=psd1D.size)
    xi = np.linspace(0, interp_num, num=interp_num)
    interpolated = griddata(points, psd1D, xi, method='cubic')

    # Normalization
    interpolated /= interpolated[0]

    return interpolated

# From https://github.com/cc-hpc-itwm/DeepFakeDetection
def fft_azimuthal_average(img, interp_num=300, epsilon=1e-8):
    img = rgb2gray(img) * 255.0

    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + epsilon)

    return azimuthal_average_2d(magnitude_spectrum, interp_num=interp_num)

# From https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4663021&tag=1
def LoG_mean_power(img, conv_mod='full', filter='paper'):
    img = rgb2gray(img) * 255.0
    if filter == 'paper':
        LoG_filter = [
            [-2, -4, -4, -4, -2],
            [-4, 0, 8, 0, -4],
            [-4, 8, 24, 8, -4],
            [-4, 0, 8, 0, -4],
            [-2, -4, -4, -4, -2]
        ]
        LoG = convolve2d(img, LoG_filter, mode=conv_mod)
    else:
        LoG = gaussian_laplace(img, sigma=.05)

    return np.array([np.mean(np.abs(LoG))])
