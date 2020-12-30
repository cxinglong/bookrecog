#!/usr/bin/env python
import sys
import os.path as osp
sys.path.insert(0, osp.dirname(__file__))

from torch import optim
from copy import deepcopy

from mmcv.runner.hooks import CheckpointHook, OptimizerHook, TextLoggerHook

from functools import partial
from mmcls.models import ResNet, GlobalAveragePooling, LinearClsHead, MobileNetv3
from deepcls.image_classifier import ImageClassifier
from deepcls.mmcls.config import make_eval_hook, DEFAULT_CONFIGS, SCHEDULE_CONFIGS

from model_datasets import *
from model_pipelines import *

def make_model(classes, tp, **kws):
    print ("using model", tp)
    if tp == 'mobilenet_v3':
        backbone = MobileNetv3('big')
        feat_chns = 112

    elif tp == 'resnet18':
        backbone = ResNet(
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch',
        )
        feat_chns = 512

    else:
        raise

    heads = {
        k: LinearClsHead(len(v), feat_chns)
        for k, v in classes.items()
    }

    net = ImageClassifier(
        backbone=backbone,
        neck=GlobalAveragePooling(),
        heads=heads
    )
    return net

def make_models(tp):
    return edict({
        tp: {
            **DEFAULT_CONFIGS['runtime'](),
            'make_model': partial(make_model, tp=tp),
            'make_img_transform': make_img_transform,

            'workers_per_gpu': 4,
            'datasets': {
                'book': {
                    **PIPELINES['default'],
                    **DATASETS['book'],
                    **SCHEDULE_CONFIGS['0.5x'],
                    'batch_size': 32,
                    'make_optimizer': partial(optim.SGD, lr=0.1, momentum=0.9, weight_decay=0.0001),
                },
            },

            'batch_size': 32,

            **SCHEDULE_CONFIGS['1x'],

            'make_optimizer': partial(optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.0001),

            'optimizer_hook': OptimizerHook(
                grad_clip={
                    'max_norm': 35,
                    'norm_type': 2,
                }
            ),

            'make_eval_hook': make_eval_hook,

            'checkpoint_hook': CheckpointHook(
                interval=2,
                max_keep_ckpts=3,
            ),
        },
    })

models = make_models('resnet18')
models.update(make_models('mobilenet_v3'))

if __name__ == "__main__":
    net = make_model({'face': ['front', 'back'], 'dir': ['up', 'down']})
