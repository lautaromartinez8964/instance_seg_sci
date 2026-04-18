from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList

from .mask_rcnn import MaskRCNN


@MODELS.register_module()
class RSMaskRCNN(MaskRCNN):
    """Mask R-CNN with optional IG-Scan foreground auxiliary loss."""

    def _set_backbone_auxiliary_targets(self, batch_inputs: Tensor,
                                        batch_data_samples: SampleList) -> None:
        if hasattr(self.backbone, 'set_auxiliary_targets'):
            self.backbone.set_auxiliary_targets(
                batch_data_samples,
                input_shape=batch_inputs.shape[-2:],
                device=batch_inputs.device)

    def _get_backbone_auxiliary_losses(self) -> dict:
        if hasattr(self.backbone, 'get_auxiliary_losses'):
            return self.backbone.get_auxiliary_losses()
        return {}

    def _clear_backbone_auxiliary_targets(self) -> None:
        if hasattr(self.backbone, 'clear_auxiliary_targets'):
            self.backbone.clear_auxiliary_targets()

    def _set_ig_targets(self, batch_inputs: Tensor,
                        batch_data_samples: SampleList) -> None:
        if hasattr(self.backbone, 'set_ig_targets'):
            self.backbone.set_ig_targets(
                batch_data_samples,
                input_shape=batch_inputs.shape[-2:],
                device=batch_inputs.device)

    def _get_ig_aux_losses(self) -> dict:
        if hasattr(self.backbone, 'get_ig_aux_losses'):
            return self.backbone.get_ig_aux_losses()
        return {}

    def _clear_ig_targets(self) -> None:
        if hasattr(self.backbone, 'clear_ig_targets'):
            self.backbone.clear_ig_targets()

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        self._set_backbone_auxiliary_targets(batch_inputs, batch_data_samples)
        self._set_ig_targets(batch_inputs, batch_data_samples)
        losses = super().loss(batch_inputs, batch_data_samples)
        losses.update(self._get_backbone_auxiliary_losses())
        losses.update(self._get_ig_aux_losses())
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        self._clear_backbone_auxiliary_targets()
        self._clear_ig_targets()
        return super().predict(batch_inputs, batch_data_samples, rescale=rescale)

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        self._clear_backbone_auxiliary_targets()
        self._clear_ig_targets()
        return super()._forward(batch_inputs, batch_data_samples)