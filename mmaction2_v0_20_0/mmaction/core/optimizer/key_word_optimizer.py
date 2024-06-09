# Copyright (c) OpenMMLab. All rights reserved.
import sys
import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd
# from generators import Bitparm

@OPTIMIZER_BUILDERS.register_module()
class KeyWordOptimizerConstructor(DefaultOptimizerConstructor):
    """Optimizer constructor in TSM model.

    This constructor builds optimizer in different ways from the default one.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and 10x weight
       decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        params.clear() ## clear the old parameters
        key_words = self.paramwise_cfg['keywords']
        trained_parameters = []
        for n,m in model.named_modules():
            if isinstance(m, _ConvNd):
                for key_word in key_words:
                    if key_word in n:
                        print("adding {} to parameters".format(n))
                        trained_parameters+= list(m.parameters())
                        break
        params.append({
            'params': trained_parameters,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

       