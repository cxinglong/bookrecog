from collections import OrderedDict
import inspect
import torch
from torch import nn
import torch.distributed as dist
from abc import ABCMeta, abstractmethod

class BaseClassifier(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward_train(self, imgs, **kws):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kws (keyword arguments): Specific to concrete implementation.
        """
        pass

    def simple_test(self, img, **kws):
        pass

    def aug_test(self, imgs, **kws):
        pass

    def forward_test(self, imgs, **kws):
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kws)
        else:
            return self.aug_test(imgs, **kws)

    def forward(self, img, return_loss=True, **kws):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, **kws)
        else:
            return self.forward_test(img, **kws)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def step(self, data, optimizer):
        has_info = 'info' in inspect.getargspec(self.forward_train).args
        if has_info:
            info = {}
            losses = self(**data, info=info)
        else:
            losses = self(**data)

        loss, log_vars = self._parse_losses(losses)

        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': len(data['img'].data),
        }

        if has_info:
            outputs.update(info)

        return outputs

    def train_step(self, *args, **kws):
        return self.step(*args, **kws)

    def val_step(self, *args, **kws):
        return self.step(*args, **kws)
