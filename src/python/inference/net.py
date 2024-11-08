# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models as models

# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}


class Flatten(nn.Module):
    """
    Flatten module attached in the model. It basically flattens the input tensor.
    """

    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        """
        flatten the input feat
        """
        return torch.flatten(feat, start_dim=self.dim)

    def flops(self, x):
        """
        number of floating point operations performed. 0 for this module.
        """
        return 0


def remove_nesting_issues(checkpoint, prefix_to_remove, prefix_to_replace):
    """
    Modify a state_dict by removing a specified prefix from all keys.

    Args:
        state_dict (dict): The state_dict to be modified.
        prefix_to_remove (str): The prefix to be removed from keys.

    Returns:
        dict: The modified state_dict with keys stripped of the specified prefix.
    """
    new_state_dict = {}
    ignored_keys = []

    try:
        for key_name in list(checkpoint.keys()).copy():
            if key_name.startswith(prefix_to_remove):
                # Remove the prefix and create a new key
                new_key = key_name.replace(prefix_to_remove, prefix_to_replace)
                new_state_dict[new_key] = checkpoint[key_name]
            else:
                print(
                    f"key name: {key_name} does not start with {prefix_to_remove} - ignoring", flush=True)
                ignored_keys.append(key_name)
    finally:
        if key_name.startswith('net.last_layers.1.'):
            new_state_dict['fc.weight'] = checkpoint['net.last_layers.1.weight']
            new_state_dict['fc.bias'] = checkpoint['net.last_layers.1.bias']
        elif key_name.startswith('net.final_layer'):
            new_state_dict['net.model.final_layer'] = checkpoint['net.final_layer']
        elif key_name.startswith('net.final_bn'):
            new_state_dict['net.model.final_bn'] = checkpoint['net.final_bn']

    return new_state_dict, ignored_keys


class EffNetV2L(nn.Module):
    def __init__(self, dropout=0, num_features=512, fp16=False, weights='IMAGENET1K_V1'):
        super(EffNetV2L, self).__init__()

        if weights is not None:
            self.model = models.efficientnet_v2_l(weights=weights)
        else:
            self.model = models.efficientnet_v2_l()

        self.num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(self.num_ftrs, num_features)

        self.features_out = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features_out.weight, 1.0)
        self.features_out.weight.requires_grad = False

        self.fp16 = fp16

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            x = self.model(x)

        # no more features out here when used as a classifier!
        # x = self.features_out(x.float() if self.fp16 else x)
        # but we still need to handle x! this can be done in the return
        return x.float() if self.fp16 else x

    def convert_to_custom_classifier(self):
        # we do not need the embedding layer for ident
        self.features_out = None
        # adjust 2nd to last layer now goes into 2 for classification
        self.model.classifier[1] = nn.Linear(self.num_ftrs, 2)
