from easydict import EasyDict as edict
from copy import deepcopy
from functools import partial

from mmcv.runner.hooks import CheckpointHook, TextLoggerHook, TensorboardLoggerHook
from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook
from mmcls.core import DistEvalHook, EvalHook
from deepcv.mmcv.hooks import ImageTensorboardHook

def make_eval_hook(data_loader, distributed, **kws):
    cls = DistEvalHook if distributed else EvalHook
    return cls(data_loader, **kws)

def make_config_runtime(log_level='INFO', ckpt_interval=10, max_keep_ckpts=5, log_interval=10, need_tensorboard=False):
    log_hooks = [TextLoggerHook(interval=log_interval)]
    if need_tensorboard:
        log_hooks.append(ImageTensorboardHook(interval=log_interval))

    return  {
        'workflow': [('train', 1), ('val', 1)],

        'log_level': log_level,

        'dist_params': {
            'backend': 'nccl',
        },

        'checkpoint_hook': CheckpointHook(
            interval=ckpt_interval,
            max_keep_ckpts=max_keep_ckpts,
        ),

        'log_hooks': log_hooks,

        'find_unused_parameters': True,
    }

DEFAULT_CONFIGS = {
    'runtime': make_config_runtime,
}

SCHEDULE_CONFIGS = {}
for i in [0.5, 1, 2, 3, 4]:
    SCHEDULE_CONFIGS[f'{i}x'] = {
        'max_epochs': int(100 * i),
        'lr_hook': StepLrUpdaterHook(
            step = [int(30*i), int(60*i), int(90*i)],
        ),
    }

def update_make_model(models, **kws):
    if 'make_model' in models:
        models['make_model'] = partial(models['make_model'], **kws)
    for v in models.datasets.values():
        if 'make_model' in v:
            v['make_model'] = partial(v['make_model'], **kws)
