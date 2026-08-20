from __future__ import annotations

import numpy as np

from tradition.core.interfaces import CFARBackend


class NumpyCACFAR(CFARBackend):
    """Independent NumPy CA-CFAR implementation in the log-power domain."""

    def __init__(self, threshold_offset_db: float = 6.0) -> None:
        self.threshold_offset_db = float(threshold_offset_db)

    def threshold(
        self,
        signal_db: np.ndarray,
        guard_len: int,
        noise_len: int,
    ) -> np.ndarray:
        signal = np.asarray(signal_db, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError("CA-CFAR backend expects a 1D signal.")
        if guard_len < 0 or noise_len <= 0:
            raise ValueError("guard_len must be >=0 and noise_len must be >0.")

        n = signal.size
        result = np.empty(n, dtype=np.float64)
        for cut in range(n):
            left_start = max(0, cut - guard_len - noise_len)
            left_end = max(0, cut - guard_len)
            right_start = min(n, cut + guard_len + 1)
            right_end = min(n, cut + guard_len + noise_len + 1)
            noise = np.concatenate(
                [signal[left_start:left_end], signal[right_start:right_end]]
            )
            result[cut] = (
                np.inf
                if noise.size == 0
                else float(noise.mean()) + self.threshold_offset_db
            )
        return result


class OpenRadarCACFAR(CFARBackend):
    """Thin adapter around the user's OpenRadar fork (`mmwave.dsp.cfar.ca_`)."""

    def __init__(self, threshold_offset_db: float = 6.0) -> None:
        try:
            from mmwave.dsp.cfar import ca_
        except ImportError as exc:
            raise ImportError(
                "OpenRadar backend requested but `mmwave` is not importable. "
                "Install the fork with `pip install -e /path/to/OpenRadar`."
            ) from exc
        self._ca = ca_
        self.threshold_offset_db = float(threshold_offset_db)

    def threshold(
        self,
        signal_db: np.ndarray,
        guard_len: int,
        noise_len: int,
    ) -> np.ndarray:
        threshold, _ = self._ca(
            np.asarray(signal_db, dtype=np.float64),
            guard_len=guard_len,
            noise_len=noise_len,
            mode="constant",
            l_bound=self.threshold_offset_db,
        )
        return np.asarray(threshold, dtype=np.float64)
