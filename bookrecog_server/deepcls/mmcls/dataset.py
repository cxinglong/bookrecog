import numpy as np
import yaml
import os
import os.path as osp
from glob import glob
import json
from mmcls.datasets import BaseDataset

class MultiClassDataset(BaseDataset):
    def __init__(self, *args, class_dic=None, tp, **kws):
        self.class_dic = class_dic
        self.tp = tp
        super().__init__(*args, **kws)

    def load_annotations(self):
        class_dic = self.class_dic
        task_nr = len(class_dic)

        folders = sorted(set(k for i in class_dic.values() for j in i.values() for k in j))

        fullnames = set()
        splits = {}
        for folder in folders:
            if folder not in splits:
                d = json.load(open(self.ann_file+f'/{folder}.json'))
                for k, v in d.items():
                    d[k] = set(v)
                splits[folder] = d

            img_dir = osp.join(self.data_prefix, folder)
            folder_fullnames = [i for i in glob(img_dir+'/*.jpg') if osp.basename(i) in splits[folder][self.tp]]

            fullnames |= set(folder_fullnames)

        tsk_id2name = {tsk_idx: tsk_name for tsk_idx, tsk_name in enumerate(class_dic)}
        cls_id2name = {}
        for tsk_name, v in class_dic.items():
            cls_id2name[tsk_name] = {
                cls_id: cls_name for cls_id, cls_name in enumerate(v)}

        data_infos = []
        for fullname in fullnames:
            folder = osp.basename(osp.dirname(fullname))
            gt_label = - np.ones(task_nr, dtype='i8')
            for tsk_idx, (tsk_name, tsk) in enumerate(class_dic.items()):
                for cls_idx, (cls_name, folders) in enumerate(tsk.items()):
                    if folder in folders:
                        gt_label[tsk_idx] = cls_idx

            fname = osp.basename(fullname)
            info = {
                'classes': class_dic,
                'tsk_id2name': tsk_id2name,
                'cls_id2name': cls_id2name,
                'img_prefix': osp.dirname(fullname),
                'img_info': {'filename': fname},
                'gt_label': gt_label,
                'txt_label': [
                    (tsk_id2name[i], cls_id2name[tsk_id2name[i]][l])
                    for i, l in enumerate(gt_label)
                ],
            }
            data_infos.append(info)

        print (f"loaded dataset {self.tp}", len(data_infos))
        return data_infos
