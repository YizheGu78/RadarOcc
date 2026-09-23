"""Streaming scene video: prediction / GT / RGB; no image accumulation."""
from pathlib import Path
import numpy as np


class Video:
    def __init__(self, path, fps=10.):
        import cv2
        self.cv2, self.path, self.frames = cv2, Path(path), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1536, 552))
        if not self.writer.isOpened():
            raise RuntimeError(f'Cannot open video writer: {path}')

    def add(self, native, gt, camera_path, token):
        cv2 = self.cv2
        # BGR: free blue, background grey, foreground red, unknown dark.
        colors = np.array([[180, 70, 20], [150, 150, 150], [30, 30, 240], [35, 35, 35]], np.uint8)
        def panel(labels, title):
            bev = np.zeros(labels.shape[:2], np.uint8)
            bev[np.all(labels == 255, axis=2)] = 3
            bev[np.any(labels == 1, axis=2)] = 1
            bev[np.any(labels == 2, axis=2)] = 2
            img = cv2.resize(colors[bev[::-1]], (512, 512), interpolation=cv2.INTER_NEAREST)
            return caption(img, title)
        def caption(img, title):
            result = np.full((552, 512, 3), 245, np.uint8)
            result[40:] = img
            cv2.putText(result, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
            return result
        rgb = cv2.imread(str(camera_path))
        if rgb is None:
            raise FileNotFoundError(f'Cannot read camera image {camera_path}')
        scale = min(512 / rgb.shape[1], 512 / rgb.shape[0])
        resized = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
        canvas = np.full((512, 512, 3), 35, np.uint8)
        h, w = resized.shape[:2]
        canvas[(512-h)//2:(512-h)//2+h, (512-w)//2:(512-w)//2+w] = resized
        self.writer.write(np.concatenate([panel(native, f'Prediction | {token}'), panel(gt, 'GT | BG grey / FG red'), caption(canvas, 'RGB | Free blue / Unknown dark')], axis=1))
        self.frames += 1

    def close(self):
        self.writer.release()
