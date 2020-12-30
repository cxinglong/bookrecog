import cv2
import albumentations as A
from mmcls.datasets.pipelines import LoadImageFromFile, Collect, ImageToTensor, Compose
from mmcls.datasets.pipelines.transforms import Normalize
from deepcv.base import imagenet_norm
from deepcls.mmcls.pipelines import AlbuAug

def make_aug_default(img_size, is_train=True, rotate_limit=10, **kws):
    imw, imh = img_size
    color_aug = []
    if is_train:
        color_aug = [
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
        ]

    geo_aug = []
    if is_train:
        geo_aug = [
            # A.HorizontalFlip(), # bad for keypoints
            A.ShiftScaleRotate(rotate_limit=rotate_limit, border_mode=cv2.BORDER_REPLICATE),
            A.Perspective((0.03, 0.05), pad_mode=cv2.BORDER_REPLICATE),
        ]

    transforms = [
        A.LongestMaxSize(max_size=imw),
        *color_aug,
        A.PadIfNeeded(min_height=imh, min_width=imw, border_mode=cv2.BORDER_REPLICATE),
        *geo_aug,
    ]

    return [
        AlbuAug(
            transforms,
            skip_img_without_ann=True,
            **kws
        )
    ]

img_augs = {
    'default': make_aug_default,
}

def make_pipelines(img_size, tsk='default', with_mask=True, with_contour=False, **test_kws):

    img_aug_kws = {
        'additional_targets': {},
    }

    pipelines = {}
    for tp in ['train', 'val']:
        pipe = [
            LoadImageFromFile(),
            *img_augs[tsk](
                img_size,
                is_train=(tp == 'train'),
                **img_aug_kws,
            ),
            Normalize(**imagenet_norm, to_rgb=True),
        ]
        pipe += [
            ImageToTensor(['img']),
            Collect(['img', 'gt_label'])
        ]
        pipelines[tp] = pipe

    return {
        **pipelines,
    }

def make_img_transform(img_size, norm_cfg=imagenet_norm, to_rgb=True, divisor=32):
    img_size = tuple(img_size)
    imw, imh = img_size
    return Compose([
        AlbuAug([
            A.LongestMaxSize(max_size=imw),
            A.PadIfNeeded(min_height=imh, min_width=imw, border_mode=cv2.BORDER_REPLICATE),

        ]),
        Normalize(**norm_cfg, to_rgb=to_rgb),
        ImageToTensor(['img']),
        Collect(['img']),
    ])

PIPELINES = {
    'default': {
        'pipelines': make_pipelines,
    },
}
