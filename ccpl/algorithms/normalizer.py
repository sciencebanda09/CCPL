"""Online state normalization using running mean/variance."""
import numpy as np


class StateNormalizer:
    def __init__(self, state_dim: int, eps: float = 1e-8, clip: float = 5.0):
        self.mean = np.zeros(state_dim, np.float64)
        self.var  = np.ones(state_dim, np.float64)
        self.count = 0
        self.eps  = eps
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.asarray(x, np.float64)
        if x.ndim == 1: x = x[None]
        batch_count = x.shape[0]
        batch_mean  = x.mean(0)
        batch_var   = x.var(0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_count / total
        self.var   = (self.var * self.count + batch_var * batch_count +
                      delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.count < 2:
            return np.clip(x, -self.clip, self.clip).astype(np.float32)
        normed = (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32) + self.eps))
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)
