'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2024 The MITRE Corporation. All Rights Reserved.
'''

from abc import ABCMeta, abstractmethod


class DetectionNetwork(metaclass=ABCMeta):
    """
    Base class for Detection Networks wrapped by this API.
    """

    @abstractmethod
    def __init__(self, name, mode="rgb", target_size=(456, 456, 3)):
        self._name = name
        self._mode = mode
        self._target_size = target_size

    @property
    def name(self):
        return self._name

    @abstractmethod
    def infer(self, fp):
        pass
