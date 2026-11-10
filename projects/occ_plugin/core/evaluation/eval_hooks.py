
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import os.path as osp
import json
import math
from numbers import Real
from pathlib import Path
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import EvalHook as BaseEvalHook


def save_box_z_metrics(runner):
    """Preserve exact validation values before TextLogger rounds its JSON log."""
    model = getattr(runner.model, 'module', runner.model)
    plugin = getattr(model, 'temporal_plugin', None)
    if not hasattr(model, 'box_z_fusion') and getattr(plugin, 'query_budget', None) is None:
        return
    row = dict(mode='val', epoch=runner.epoch+1)
    row.update({k: float(v) if math.isfinite(v) else None
                for k, v in runner.log_buffer.output.items()
                if k.startswith(('SC', 'SSC')) and isinstance(v, Real)})
    if 'SSC_mean' not in row:
        return
    path = Path(runner.work_dir)/'val_metrics.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


class OccEvalHook(BaseEvalHook):
    def __init__(self, *args,  **kwargs):
        super(OccEvalHook, self).__init__(*args, **kwargs)  
            
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from projects.occ_plugin.occupancy.apis.test import custom_single_gpu_test
        results = custom_single_gpu_test(runner.model, self.dataloader, show=False)
        
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        save_box_z_metrics(runner)
        if self.save_best:
            self._save_ckpt(runner, key_score)
            
            
class OccDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args,  **kwargs):
        super(OccDistEvalHook, self).__init__(*args, **kwargs)       

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.occ_plugin.occupancy.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            
            key_score = self.evaluate(runner, results)
            save_box_z_metrics(runner)

            if self.save_best:
                self._save_ckpt(runner, key_score)
  
