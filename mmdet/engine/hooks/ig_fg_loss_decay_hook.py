from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class IGFGLinearDecayHook(Hook):
    """Linearly decay IG foreground auxiliary loss weight by epoch."""

    def __init__(self,
                 start_weight: float,
                 end_weight: float,
                 begin_epoch: int = 0,
                 end_epoch: int = 12):
        self.start_weight = float(max(start_weight, 0.0))
        self.end_weight = float(max(end_weight, 0.0))
        self.begin_epoch = int(begin_epoch)
        self.end_epoch = int(end_epoch)
        if self.end_epoch < self.begin_epoch:
            raise ValueError('end_epoch must be >= begin_epoch')

    def _unwrap_model(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model

    def _get_weight(self, epoch: int) -> float:
        if epoch <= self.begin_epoch:
            return self.start_weight
        if epoch >= self.end_epoch:
            return self.end_weight
        span = max(self.end_epoch - self.begin_epoch, 1)
        ratio = (epoch - self.begin_epoch) / span
        return self.start_weight + ratio * (self.end_weight - self.start_weight)

    def before_train_epoch(self, runner) -> None:
        model = self._unwrap_model(runner)
        backbone = getattr(model, 'backbone', None)
        if backbone is None or not hasattr(backbone, 'set_ig_fg_loss_weight'):
            return
        weight = self._get_weight(runner.epoch)
        backbone.set_ig_fg_loss_weight(weight)
        runner.logger.info(
            f'[IG-FG-Decay] epoch={runner.epoch + 1}, fg_loss_weight={weight:.4f}')