import os
import torch
import torch.multiprocessing as mp
from torch import distributed as dist

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, dev_ids=[], **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    if dev_ids:
        dev_id = dev_ids[rank % len(dev_ids)]
    else:
        num_gpus = torch.cuda.device_count()
        dev_id = rank % num_gpus

    torch.cuda.set_device(dev_id)
    dist.init_process_group(backend=backend, **kwargs)
