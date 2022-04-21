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


def CosmeticLensNetwork(img_shape, num_features, pretrain_weights=False, freeze_pretrained=False):
    embedded_model = efficientnetb5_network(img_shape, pretrain_weights=pretrain_weights, freeze_pretrained=freeze_pretrained)
    return ClassifierHead(embedded_model, img_shape, num_features)


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


# Single dense -> softmax layer outputing classifier predictions
def ClassifierHead(embed_network, img_shape, num_classes):
    input = Input(shape=img_shape)
    x = embed_network(input)
    top = Dense(num_classes, activation='softmax', name="output_classes")(x)
    return Model(inputs=input, outputs=top)


def efficientnetb5_network(img_shape, pretrain_weights=False, freeze_pretrained=False):
    from tensorflow.keras.applications.efficientnet import EfficientNetB5
    if pretrain_weights:
        weights = 'imagenet'
    else:
        weights = None

    base_model = EfficientNetB5(
        input_shape=img_shape,
        weights=weights,
        include_top=False,
        pooling='avg'
    )

    if pretrain_weights:
        if freeze_pretrained:
            base_model.trainable = False
        else:
            base_model.trainable = True

    return Model(inputs=base_model.input, outputs=base_model.output)

