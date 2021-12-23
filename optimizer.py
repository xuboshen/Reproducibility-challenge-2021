import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn

def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.lr_scheduler
    stepsize = optim_cfg.stepsize
    gamma = optim_cfg.gamma_for_lr
    max_epoch = optim_cfg.max_epoch

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

#     if optim_cfg.warmup_epoch > 0:
#         if not optim_cfg.WARMUP_RECOUNT:
#             scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

#         if optim_cfg.WARMUP_TYPE == 'constant':
#             scheduler = ConstantWarmupScheduler(
#                 optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
#                 optim_cfg.WARMUP_CONS_LR
#             )

#         elif optim_cfg.WARMUP_TYPE == 'linear':
#             scheduler = LinearWarmupScheduler(
#                 optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
#                 optim_cfg.WARMUP_MIN_LR
#             )

#         else:
#             raise ValueError
    return scheduler

def build_optimizer(model, optim_cfg):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """
    optim = optim_cfg.optimizer
    lr = optim_cfg.lr
    weight_decay = 5e-5
    momentum = 0.9
    sgd_nesterov = False
    adam_beta1 = optim_cfg.adam_beta1
    adam_beta2 = 0.999
    staged_lr = False

    if isinstance(model, nn.Module):
        param_groups = model.parameters()
    else:
        param_groups = model

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            # dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )
    return optimizer

## scheduler construction
class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]

