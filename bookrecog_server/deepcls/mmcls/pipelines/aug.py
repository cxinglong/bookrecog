import numpy as np
import numpy.random as npr
import cv2
import time
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations import DualTransform, ImageOnlyTransform

class CoarseDropoutImageOnly(A.CoarseDropout):
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply(self, image, fill_value=0, **kws):
        if isinstance(fill_value, tuple):
            fill_value = npr.randint(*fill_value)

        return super().apply(image, fill_value, **kws)

class InferAlbuAug:
    def __init__(self, transforms):
        self.transforms = transforms
        self.aug = A.Compose(self.transforms)

    def __call__(self, inp):
        out = inp.copy()
        out['img'] = self.aug(image=inp['img'])['image']
        return out

class AlbuAug:
    def __init__(
            self,
            transforms,
            skip_img_without_ann=False,
            update_pad_shape=True,
            additional_targets={}
    ):

        self.transforms = transforms
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_ann = skip_img_without_ann

        self.aug = A.Compose(
            self.transforms,
        )

    def __call__(self, inp):
        aug_input = {'image': inp['img']}

        inst_nr = 0
        has_labels = 'gt_labels' in inp
        if has_labels:
            inst_nr = len(inp['gt_labels'])

        out = inp.copy()
        trans_d = self.aug(**aug_input)
        out['img'] = trans_d['image']
        if self.update_pad_shape:
            out['pad_shape'] = out['img'].shape

        return out

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str
