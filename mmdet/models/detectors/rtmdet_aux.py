# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList

from .rtmdet import RTMDet


@MODELS.register_module()
class RTMDetWithAuxNeck(RTMDet):
    """RTMDet variant that collects auxiliary losses from the neck."""

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        if self.with_neck and hasattr(self.neck, 'set_auxiliary_targets'):
            self.neck.set_auxiliary_targets(
                batch_data_samples,
                input_shape=tuple(batch_inputs.shape[-2:]),
                device=batch_inputs.device)

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)

        losses = self.bbox_head.loss(x, batch_data_samples)
        if self.with_neck and hasattr(self.neck, 'get_auxiliary_losses'):
            losses.update(self.neck.get_auxiliary_losses())
        return losses