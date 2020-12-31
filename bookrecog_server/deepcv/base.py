import os.path as osp
import importlib.util
from functools import partial
from easydict import EasyDict as edict

import numpy as np
import torch
from skimage.color import gray2rgb

def file_name(fname):
    return osp.splitext(osp.basename(fname))[0]

BACKGROUND_ID = -1
IGNORE_ID = -2

def to_rgb(img):
    if img.ndim == 2:
        img = gray2rgb(img)
    return img

def make_class_dic(class_names):
    classes = {
        'ignore': {'name': 'ignore', 'id': IGNORE_ID},
        'background': {'name': 'background', 'id': BACKGROUND_ID},
    }
    classes.update({idx: {
        'name': name,
        'id': idx
    } for idx, name in enumerate(class_names)})
    return classes

def get_type(v):
    if isinstance(v, dict):
        return v['type']

    if hasattr(v, 'type') and isinstance(v.type, str):
        return v.type
    return v.__class__.__name__

def th_val(v, dtype=None):
    if isinstance(v, np.ndarray):
        return v

    v = v.detach().cpu().numpy()
    if dtype:
        v = v.astype(dtype)
    return v

def scale_grad(t, factor):
    assert 0 <= factor <= 1
    return (1 - factor) * t.detach() + factor * t

def get_gt_masks(masks):
    if hasattr(masks, 'masks'):
        masks = masks.masks
    return masks

def denorm_img_by_meta(img, img_metas):
    # for visualization e.g. tensorboard
    means = [torch.Tensor(i['img_norm_cfg']['mean']).to(img.device) for i in img_metas]
    stds = [torch.Tensor(i['img_norm_cfg']['std']).to(img.device) for i in img_metas]
    mean = torch.stack(means, dim=0).view(-1, 3, 1, 1)
    std = torch.stack(stds, dim=0).view(-1, 3, 1, 1)
    return (img * std + mean) / 255.0

# def check_boxes(boxes, fname='debug_box.pkl'):
#     boxes = th_val(boxes)
#     diff = boxes[..., 2:] >= boxes[...,:2]
#     cond = diff.all()
#     if not cond:
#         import os
#         import os.path as osp
#         print (fname)
#         print (boxes.shape)
#         sel = np.logical_and(diff[..., 0], diff[..., 1])
#         print (sel)
#         print (boxes[~sel, ...])
#         import pickle
#         os.makedirs('debug', exist_ok=True)
#         pickle.dump(boxes, open(osp.join('debug', fname),'wb'))

#     assert cond

imagenet_norm = edict({
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375],
})

def import_module_from_file(fname):
    module_name = file_name(fname)
    spec = importlib.util.spec_from_file_location(module_name, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def import_module_and_key(fname, key=None):
    parts = fname.split(':')
    module_f = parts[0]
    if not key and len(parts) <= 1:
        key = 'default'

    if not key:
        key = parts[-1]

    m = import_module_from_file(module_f)
    return m, key

def import_model(model_f, ds=None):
    m, net_type = import_module_and_key(model_f)
    cfg = m.models[net_type]
    if ds:
        cfg.update(cfg.datasets[ds])

    cfg.net_type = net_type
    cfg.ds = ds
    return cfg

def load_dataset(cfg, data_root, pipelines=None, tp=None):
    if pipelines is None:
        assert 'pipelines' in cfg
        pipelines = cfg.pipelines

    if tp is None:
        tp = 'train'
    img_size = cfg['img_size']
    data_root = osp.join(data_root, cfg['data_root']).format(IMG_WIDTH=img_size[0], IMG_HEIGHT=img_size[1])
    if isinstance(pipelines, dict) and 'pipelines' in pipelines:
        pipelines = pipelines['pipelines']
    return cfg.make_dataset(data_root, tp, pipelines=pipelines(img_size), img_size=img_size, classes=cfg['classes'])

def import_dataset(dataset_f, pipeline_f, data_root, ds=None, tp=None, pipeline=None):
    m_ds, ds  = import_module_and_key(dataset_f, ds)
    m_p, pipeline = import_module_and_key(pipeline_f, pipeline)
    return load_dataset(m_ds.DATASETS[ds], data_root, pipelines=m_p.PIPELINES[pipeline], tp=tp)

def tensor2img(v, norm=None):
    if norm is None:
        norm = imagenet_norm
    v = th_val(v)
    v = v.transpose((1, 2, 0))
    v = (v * norm['std'] + norm['mean']).astype('u1')
    return v

def denorm_sample_img(s, idx=0):
    img = s['img'][idx]
    norm_cfg = s['img_metas'][0].data['img_norm_cfg']
    mean, std = norm_cfg['mean'], norm_cfg['std']
    img = th_val(img).transpose((1, 2, 0))
    return (img * std + mean).astype('u1')

def szip(*args):
    assert all(len(i) == len(args[0]) for i in args[1:])
    return zip(*args)
