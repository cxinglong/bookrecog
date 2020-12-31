import os.path as osp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
from mmcv.runner import master_only
from mmcv.runner.hooks import TensorboardLoggerHook

class ImageTensorboardHook(TensorboardLoggerHook):
    def __init__(self, flush_secs=10, **kws):
        super().__init__(**kws)
        self.flush_secs = flush_secs

    @master_only
    def before_run(self, runner):
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tb_logs', datetime.now().strftime('%y%m%dT%H%M%S'))
        self.writer = SummaryWriter(self.log_dir, flush_secs=self.flush_secs)

    @master_only
    def log(self, runner):
        super().log(runner)

        data = runner.outputs
        log_images = data.get('log_images', {})
        for k, v in log_images.items():
            grid = tv.utils.make_grid(v)
            self.writer.add_image(k, grid, runner.iter)
