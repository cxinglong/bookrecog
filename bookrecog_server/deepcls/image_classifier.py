from torch import nn
from .base_classifier import BaseClassifier

class ImageClassifier(BaseClassifier):
    def __init__(self, heads, backbone, neck=None, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = nn.ModuleDict(heads)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        if self.neck is not None:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        for v in self.heads.values():
            v.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kws):
        x = self.extract_feat(img)

        results = {
            key: head.forward_train(x, gt_label[:, idx])
            for idx, (key, head) in enumerate(self.heads.items())
        }

        out = {}
        for tsk, v in results.items():
            out[f'loss_{tsk}'] = v['loss']
            for kk, vv in v['accuracy'].items():
                out[f'acc_{tsk}_{kk}'] = vv

        return out

    def simple_test(self, img):
        x = self.extract_feat(img)

        rets = {
            key: head.simple_test(x)
            for idx, (key, head) in enumerate(self.heads.items())
        }

        outs = [{k: v[i] for k, v in rets.items()}
                for i in range(len(img))]

        return outs
