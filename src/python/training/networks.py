'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization, \
    MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, LeakyReLU, Lambda
import sys
sys.path.append('../inference')
from cv_features.tf_features import FFTAzimuthalAverage, BSIFFeatures


def CosmeticLensNetwork(img_shape, num_features, drop_out=0.25, num_hiddens=32,
                   num_residual_layers=6, num_residual_hiddens=[64, 128, 256, 512]):
    """

    :param img_shape:
    :param num_features:
    :param drop_out:
    :param num_hiddens:
    :param num_residual_layers:
    :param num_residual_hiddens:
    :return:
    """
    resolution_log2 = int(np.log2(img_shape[0]))
    # target_log2 = 5         # 32 x 32
    # target_log2 = 4  # 16 x 16, but reaches only 20x20
    target_log2 = 3  # 8 x 8, but reaches only 10x10

    def res_stack(h):
        name = "res_stack"
        for s in range(len(num_residual_hiddens)):
            for i in range(num_residual_layers):
                h2 = h
                h2 = Conv2D(num_residual_hiddens[s], 1, padding='same', activation='relu', name=f"{name}-{s}-{i}-conv2d-1")(h2)
                h2 = Conv2D(num_residual_hiddens[s], 3, padding='same', activation='relu', name=f"{name}-{s}-{i}-convd2-2")(h2)
                h2 = Conv2D(num_hiddens, 1, padding='same', activation='relu', name=f"{name}-{s}-{i}-convd2-3")(h2)
                h = Add()([h, h2])
            h = BatchNormalization()(h)
            h = Dropout(drop_out)(h)
        return h

    input = Input(shape=img_shape)
    x = Conv2D(num_hiddens // 2, 7, padding='same',  activation='relu')(input)
    # decrease H and W
    for r in range(resolution_log2, target_log2, -1):
        x = Conv2D(num_hiddens, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D()(x)

    # do res network, should be  [N, 10, 10, num_hiddens] -> (10 x 10 x 64)
    x = res_stack(x)
    x = Conv2D(num_hiddens, 1, padding='same', activation='relu', name=f"Conv2D-Res-End")(x)
    x = Dropout(drop_out)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hiddens, activation='relu', name="cnn_dense")(x)

    # do fft
    f = FFTAzimuthalAverage()(input)
    # do bsif
    b = BSIFFeatures()(input)
    x = Concatenate()([x, f, b])
    x = Dense(num_hiddens, activation='relu', name="combined_dense")(x)
    top = Dense(num_features, activation='softmax', name="output_classes")(x)

    return Model(inputs=input, outputs=top, name="simple_network")

def SoftLensNetwork(img_shape, num_features, num_hiddens=32, pretrain_weights=True, freeze_pretrained=False):
    # see https://blog.godatadriven.com/rod-keras-multi-label
    from tensorflow.keras.applications.resnet50 import ResNet50
    if pretrain_weights:
        weights = 'imagenet'
    else:
        weights = None

    base_model = ResNet50(input_shape=img_shape,
                       weights=weights,
                       include_top=False)

    if pretrain_weights:
        if freeze_pretrained:
            base_model.trainable = False
        else:
            base_model.trainable = True

    x = base_model.output   # conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0]
    x = GlobalAveragePooling2D()(x)
    
    # do fft
    f = FFTAzimuthalAverage()(base_model.input)
    # do bsif
    b = BSIFFeatures()(base_model.input)
    x = Concatenate()([x, f, b])
    x = Dense(num_hiddens, activation='relu', name="combined_dense")(x)
    top = Dense(num_features, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=top)

