import os.path as osp
import time

import torch
import mmcv
from mmcv.runner import load_checkpoint, save_checkpoint, get_dist_info, LogBuffer, get_time_str, get_host_info, Hook, get_priority

class Trainer:
    def __init__(self, model, optimizer=None, work_dir=None,
                 logger=None, meta=None, max_iters=None, max_epochs=None):

        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta

        if work_dir:
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)

        elif work_dir is None:
            self.work_dir = None

        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self.model_name = self.model.module.__class__.__name__
        else:
            self.model_name = self.model.__class__.__name__

        self.rank, self.world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None
        self.hooks = []
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0

        self.max_epochs = max_epochs
        self.max_iters = max_iters

        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()

    def run_iter(self, data_batch, train_mode, **kwargs):
        if train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kws):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.max_iters = self.max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')

        time.sleep(0.5)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kws)
            self.call_hook('after_train_iter')
            self.iter += 1

        self.call_hook('after_train_epoch')
        self.epoch += 1

    def val(self, data_loader, **kws):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(0.5)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False, **kws)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, **kws):
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        assert self.max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self.max_iters = self.max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self.max_epochs)
        self.call_hook('before_run')

        while self.epoch < self.max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self.max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kws)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):

        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)

        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            mmcv.symlink(filename, dst_file)

    def current_lr(self):
        """Get current learning rates.
        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def current_momentum(self):
        """Get current momentums.
        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        if not hook:
            return

        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self.hooks) - 1, -1, -1):
            if priority >= self.hooks[i].priority:
                self.hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self.hooks.insert(0, hook)

    def call_hook(self, fn_name):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self.epoch = checkpoint['meta']['epoch']
        self.iter = checkpoint['meta']['iter']

        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
