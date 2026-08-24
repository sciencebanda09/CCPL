import numpy as np
from dataclasses import dataclass
 
 
@dataclass
class Transition:
    state:       np.ndarray
    action:      int
    reward:      float
    next_state:  np.ndarray
    consequence: float
    done:        bool
    hidden:      np.ndarray
    next_hidden: np.ndarray
    td_error:    float = 1.0
    c_error:     float = 1.0
    uncertainty: float = 0.5
 
    @property
    def priority(self):
        return abs(self.td_error) + abs(self.c_error) + self.uncertainty + 1e-6
 
 
class ReplayBuffer:
    def __init__(self, capacity=50_000, alpha_p=0.6, beta_p=0.4,
                 w_td=0.5, w_c=0.3, w_u=0.2, seed=42,
                 gru_dim: int = 40, beta_frames: int = 100_000):
        self.capacity = capacity
        self.alpha_p  = alpha_p
        self.beta_p   = beta_p
        self._beta_start = float(beta_p)
        self._beta_frames = max(int(beta_frames), 1)
        self._total_pushes = 0
        self.w_td     = w_td
        self.w_c      = w_c
        self.w_u      = w_u
        self.gru_dim  = gru_dim
        self.rng      = np.random.default_rng(seed)
        self._buf: list[Transition] = []
        self._pos = 0
 
    def push(self, state, action, reward, next_state, consequence, done,
             hidden=None, next_hidden=None):
        h  = (np.zeros(self.gru_dim, np.float32)
              if hidden is None else np.asarray(hidden).squeeze())
        nh = (np.zeros(self.gru_dim, np.float32)
              if next_hidden is None else np.asarray(next_hidden).squeeze())
        t  = Transition(
            np.array(state,      np.float32), int(action), float(reward),
            np.array(next_state, np.float32), float(consequence), bool(done), h, nh,
        )
        if len(self._buf) < self.capacity:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
        self._pos = (self._pos + 1) % self.capacity
        self._total_pushes += 1
        fraction = min(self._total_pushes / self._beta_frames, 1.0)
        self.beta_p = self._beta_start + fraction * (1.0 - self._beta_start)
 
    def _normalized_priorities(self, buf):
        td  = np.array([abs(t.td_error)                          for t in buf], np.float32) + 1e-6
        ce  = np.array([abs(getattr(t, "c_error",     t.td_error)) for t in buf], np.float32) + 1e-6
        unc = np.array([abs(getattr(t, "uncertainty", 0.5))       for t in buf], np.float32) + 1e-6
 
        td  /= td.max()
        ce  /= ce.max()
        unc /= unc.max()
 
        return (self.w_td * td + self.w_c * ce + self.w_u * unc) ** self.alpha_p
 
    def sample(self, batch_size):
        eligible = self._buf
        if len(eligible) < batch_size:
            return None
 
        priorities = self._normalized_priorities(eligible)
 
        priorities = np.clip(priorities, 1e-8, None)
        probs      = priorities / priorities.sum()
 
        idxs    = self.rng.choice(len(eligible), batch_size, replace=True, p=probs)
        n       = len(eligible)
        weights = (n * probs[idxs]) ** (-self.beta_p)
        weights /= weights.max()
 
        batch = [eligible[i] for i in idxs]
        return {
            "states":       np.stack([b.state       for b in batch]),
            "actions":      np.array([b.action      for b in batch], np.int32),
            "rewards":      np.array([b.reward      for b in batch], np.float32),
            "next_states":  np.stack([b.next_state  for b in batch]),
            "consequences": np.array([b.consequence for b in batch], np.float32),
            "dones":        np.array([b.done        for b in batch], np.float32),
            "hiddens":      np.stack([b.hidden      for b in batch]),
            "next_hiddens": np.stack([b.next_hidden for b in batch]),
            "weights":      weights.astype(np.float32),
            "indices":      idxs,
        }
 
    def update_priorities(self, indices, td_errors, c_errors=None, uncertainties=None):
        for i, idx in enumerate(indices):
            if idx < len(self._buf):
                t          = self._buf[idx]
                t.td_error = abs(float(td_errors[i])) + 1e-6
                if c_errors      is not None: t.c_error     = abs(float(c_errors[i]))      + 1e-6
                if uncertainties is not None: t.uncertainty = abs(float(uncertainties[i])) + 1e-6
 
    def ready_count(self): return len(self._buf)
    def __len__(self):     return len(self._buf)
