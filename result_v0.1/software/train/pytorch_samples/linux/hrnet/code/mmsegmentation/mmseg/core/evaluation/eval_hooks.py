import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import os, sys
import time

class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.iters = 0

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        flag_done = torch.tensor(0.0).cuda()
        train_done = self.evaluate(runner, results)
        dist.barrier()
        if train_done and runner.rank == 0:
            flag_done = torch.tensor(1.0).cuda()
            start_time = float(os.environ['START_TIME'])
            print('AI-Rank-log %s target_quality_time:%.2fsec' % (time.time(), time.time() - start_time))
            global batch_size
            batch_size = int(os.environ['BATCH_SIZE'])
            print('AI-Rank-log %s test_finish' % (time.time()))
            print('AI-Rank-log %s total_use_time:%ssec' % (time.time(), time.time() - start_time))
            print('AI-Rank-log %s avg_ips:%simages/sec' % (time.time(), self.iters * batch_size / (time.time() - start_time)))
        dist.broadcast(flag_done, 0)
        if flag_done.item() > 0.5:
            sys.exit('training done')


    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        self.iters += self.interval
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        train_done = False
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
            if name == 'mIoU':
                # global train_data_len
                train_data_len = int(os.environ['TRAIN_DATA_LEN'])
                batch_size = int(os.environ['BATCH_SIZE'])
                iters_every_epoch = train_data_len / batch_size
                print('AI-Rank-log %s eval_accuracy:%s, total_epoch_cnt:%.2f' % (time.time(), val*100, self.iters/iters_every_epoch))
            try:
                TopVal = float(os.environ['TOP_VAL'])
            except:
                TopVal = 78.5
            if name == 'mIoU' and val * 100 >= TopVal:
                train_done = True
                
        runner.log_buffer.ready = True
        return train_done        


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.iters = 0

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        flag_done = torch.tensor(0.0).cuda()
        if runner.rank == 0:
            print('\n')
            train_done = self.evaluate(runner, results)
            if train_done:
                flag_done = torch.tensor(1.0).cuda()
                global start_time
                start_time = float(os.environ['START_TIME'])
                print('AI-Rank-log %s target_quality_time:%.2fsec' % (time.time(), time.time() - start_time))
                global batch_size
                batch_size = int(os.environ['BATCH_SIZE'])
                print('AI-Rank-log %s test_finish' % (time.time()))
                print('AI-Rank-log %s total_use_time:%ssec' % (time.time(), time.time() - start_time))
                print('AI-Rank-log %s avg_ips:%simages/sec' % (time.time(), self.iters * batch_size / (time.time() - start_time)))
        dist.broadcast(flag_done, 0)
        dist.barrier()
        if flag_done.item() > 0.5:
            exit()
    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)


