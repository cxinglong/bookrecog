import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))

import yaml
from easydict import EasyDict as edict
from deepcls.mmcls.dataset import MultiClassDataset

def get_pipeline(pipelines, tp):
    if 'train' in tp:
        p = pipelines['train']
    elif 'val' in tp:
        p = pipelines['val']
    else:
        assert tp == 'test'
        p = pipelines['test']
    return p

def make_dataset_book(data_root, tp, pipelines, classes, img_size):
    assert tp in ('train', 'val_train', 'val', 'test')
    return  MultiClassDataset(
        class_dic=classes,
        ann_file=osp.join(data_root, 'splits'),
        data_prefix=osp.join(data_root, f'images_{img_size[0]}x'),
        pipeline=get_pipeline(pipelines, tp),
        tp=tp,
    )


book_class_dic = yaml.safe_load(open(osp.join(cur_dir, 'tasks.yaml')))['class_dic']

DATASETS = edict({
    'book': {
        'classes': book_class_dic,
        'data_root': 'books/',
        'img_size': (512, 512),
        'make_dataset': make_dataset_book,
    }
})

if __name__ == "__main__":
    ds = make_dataset_book('../data/books', 'train', {'train': []}, {}, (1024, 1024))

    ds.load_annotations()
