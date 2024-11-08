import os
import sys

from PIL import Image

import torch
from torchvision import transforms as T
import pytorch_lightning as pl

sys.path.append(os.environ['BIQT_HOME'] + '/providers/BIQTContactDetector/src/python/inference')
from net import EffNetV2L

class IrisDetector(pl.LightningModule):
    def __init__(self):

        super().__init__()

        # will default to 'IMAGENET1K_V1' before loading a state dict if provided
        self.net = EffNetV2L(num_features=512, fp16=True, weights=None)

        if torch.cuda.is_available():
            self.net = self.net.cuda()

        # if the last layer is not configured to output a linear of 2, we must update it
        if self.net.model.classifier[-1].out_features != 2:
            # now update the end of the model if necessary
            self.net.convert_to_custom_classifier()

        self.net.eval()

    def forward(self, x):
        return self.net(x)


class IrisDetection():
    def __init__(self, cdm_ckpt=None):
        self.img_size = 480

        self.cdm = IrisDetector.load_from_checkpoint(cdm_ckpt)
        self.softmax = torch.nn.Softmax(dim=1)

        self.transform_cropped_blue_only_w_imnet_norm = T.Compose([
            T.CenterCrop(self.img_size),
            # No Aug
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def infer(self, img_path):
        # Inference

        # Load and make sure image has 3 channels
        img = Image.open(img_path).convert('RGB')

        # Center crop and normalize
        img = self.transform_cropped_blue_only_w_imnet_norm(img)

        if torch.cuda.is_available():
            img = img.cuda()

        # Forward pass
        unsqueezed_img = img.unsqueeze(0)
        cosmetic_pred = self.cdm(unsqueezed_img)

        # Apply softmax to obtain values [0,1] for each prediction
        cosmetic_softmax_pred = self.softmax(cosmetic_pred)[:, 1].item()

        return cosmetic_softmax_pred
