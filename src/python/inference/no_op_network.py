'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2024 The MITRE Corporation. All Rights Reserved.
'''

import logging

from detection_network import DetectionNetwork


class NoOpNetwork(DetectionNetwork):

    """
    The NoOpNetwork is a DetectionNetwork which returns scores of 0.05 for all inputs. It is
    used for testing.
    """

    def __init__(self):
        logging.debug("Initializing a No-Op Network...")
        super().__init__("noop")

    def infer(self, fp):
        logging.info(f"infer fp")
        return {
            "bad_iris_score": 0.05,
            "cosmetic_score": 0.05
        }
