# -*- coding: utf-8 -*-
"""
Subject disjoint combined NDCLD13 and NDCLD15 datasets

"""

import os
import os.path as osp
import pandas as pd

# should create base class
class GenericDataset:

    def __init__(self, train_metadata_file, validation_metadata_file, test_metadata_file=None,
                images_folder=None, suffix='', img_path_field='Image'):
        """

        :param main_folder:
        :param suffix:
        :param binarize:
        """
        self.__suffix = suffix
        self.img_path_field = img_path_field
        if images_folder is None:
            self.images_folder = ''
        else:
            self.images_folder = images_folder

        self.test_path = None
        self.train_path = train_metadata_file
        self.val_path = validation_metadata_file
        if test_metadata_file is not None:
            self.test_path = test_metadata_file

        self.features_name = []
        self.files = "Image_Path"                           # column holding path to image: aka, x_col
        self.__prepare()

    def __prepare(self):
        """do some pre-processing before using the data: e.g. feature selection"""
        # attributes:
        self.train_attributes = None
        self.val_attributes = None
        self.test_attributes = None
        self.train_attributes = pd.read_csv(self.train_path)
        self.val_attributes = pd.read_csv(self.val_path)
        if self.test_path is not None:
            self.test_attributes = pd.read_csv(self.test_path)

        self.train_attributes['Image_Path'] = self.train_attributes.apply(
            lambda row: os.path.join(self.images_folder, row[self.img_path_field] + self.__suffix), axis=1)

        self.val_attributes['Image_Path'] = self.val_attributes.apply(
            lambda row: os.path.join(self.images_folder, row[self.img_path_field] + self.__suffix), axis=1)

        if self.test_attributes is not None:
            self.test_attributes['Image_Path'] = self.test_attributes.apply(
                lambda row: os.path.join(self.images_folder, row[self.img_path_field] + self.__suffix), axis=1)

        self.features_name = list(set(self.val_attributes.columns) - {"Image", "Image_File", "Image_Path"})

    def split(self, name='training'):
        """Returns the ['training', 'validation', 'all'] split of the dataset"""

        # select split:
        if name == 'training':
            if self.train_attributes.empty:
                raise ValueError("No training data was loaded")
            return self.train_attributes
        elif name == 'validation':
            if self.val_attributes.empty:
                raise ValueError("No validation data was loaded")
            return self.val_attributes
        elif name == 'testing':
            if self.test_attributes is None:
                raise ValueError("No test set available for this dataset")
            elif self.test_attributes.empty:
                raise ValueError("No testing data was loaded")
            return self.test_attributes
        elif name == 'all':
            return pd.concat(self.val_attributes, self.train_attributes)
        else:
            raise ValueError('`name` must be one of [training, validation, testing]')
