import torch
from torch import nn
import numpy as np
from functools import partial
from .base import th_val

class ParallelWrapper(nn.Module):
    def __init__(self, model, rescale=True):
        super().__init__()
        self.model = model
        self.model.eval()
        self.fwd_kws = {}
        if rescale is not None:
            self.fwd_kws = {
                'rescale': rescale,
            }

    def forward(self, input):
        return self.model(**input, return_loss=False, **self.fwd_kws)

def make_parallel_model(model, dev_ids, **kws):
    ref_dev_id = dev_ids[0]
    model = model.to(ref_dev_id)
    m = ParallelWrapper(model, **kws)
    m.dev_ids = dev_ids
    pmodel = nn.parallel.replicate(m, dev_ids)
    return pmodel

def make_sample(img, filename='', **kws):
    return {
        'img': img,
        'img_shape': img.shape,
        'ori_shape': img.shape,
        'filename': filename,
        'ori_filename': filename,
        'flip': False,
        'flip_direction': '',
        **kws,
    }

def dc_to_dict(v):
    if hasattr(v, 'data'):
        v = v.data
    assert isinstance(v, dict)
    return v

def make_batch(samples, dev_id):
    img_metas = []
    imgs = []
    for s in samples:
        if 'img_metas' in s:
            assert len(s['img_metas']) == 1
            img_metas.append(dc_to_dict(s['img_metas'][0]))

        img = s['img']
        if img.ndim == 4:
            img = img[0]
        imgs.append(img)

    imgs = torch.stack(imgs, dim=0).to(dev_id)
    out = {
        'img': [imgs],

    }
    if img_metas:
        out['img_metas'] = [img_metas]

    return out

def to_device(v, dev=None):
    if dev is None:
        return v
    if isinstance(v, torch.Tensor):
        if dev == 'np':
            return th_val(v)
        assert isinstance(dev, int) or dev == 'cpu'
        return v.to(dev)

    elif 'Instances' in str(type(v)):
        if dev == 'np':
            dev = 'cpu'

        assert isinstance(dev, int) or dev == 'cpu'
        return v.to(dev)

    return v

def rec_map(l, f):
    for v in l:
        if isinstance(v, list):
            yield list(rec_map(v, f))
        elif isinstance(v, tuple):
            yield tuple(rec_map(v, f))
        else:
            yield f(v)

def parallel_infer(pmodel, imgs, transform, batch_per_gpu=1, img_metas=[], to_dev_id='first', info=None):
    dev_ids = pmodel[0].dev_ids
    if to_dev_id == 'first':
        to_dev_id = dev_ids[0]

    if not img_metas:
        img_metas = [{}] * len(imgs)

    dev_nr = len(dev_ids)
    step = batch_per_gpu * dev_nr

    samples = [make_sample(i, **img_meta)
               for idx, (i, img_meta)  in enumerate(zip(imgs, img_metas))]

    samples = [transform(s) for s in samples]

    if isinstance(info, dict):
        info['samples'] = samples

    psamples = []
    for i in range(int(np.ceil(len(samples) / step))):
        gpus_samples = samples[i*step:(i+1)*step]

        batch = []
        for j in range(int(np.ceil(len(gpus_samples) / batch_per_gpu))):
            gpu_samples = gpus_samples[j*batch_per_gpu:(j+1)*batch_per_gpu]
            dev = dev_ids[j]
            batch.append(make_batch(gpu_samples, dev))

        psamples.append(batch)

    to_dev_fn = partial(to_device, dev=to_dev_id)

    out = []
    with torch.no_grad():
        for ps in psamples:
            assert len(pmodel) >= len(ps)
            pout = nn.parallel.parallel_apply(pmodel[:len(ps)], ps)
            pout = list(rec_map(pout, to_dev_fn))

            for i in pout:
                out += i

    return out
