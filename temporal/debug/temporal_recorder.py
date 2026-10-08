"""Small offline geometry recorder; never renders inside forward()."""
from pathlib import Path
import numpy as np


class TemporalRecorder:
    def __init__(self, output_dir, save_every=50):
        if int(save_every) < 1:
            raise ValueError("save_every must be positive")
        self.output_dir = Path(output_dir)
        self.save_every = int(save_every)
        self.step = 0

    def record(self, **payload):
        step = self.step
        self.step += 1
        if step % self.save_every:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_dir / f"alignment_{step:06d}.npz", **payload)
