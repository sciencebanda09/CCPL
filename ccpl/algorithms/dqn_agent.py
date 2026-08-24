"""DQN and Double-DQN agents — full gradient backprop through trunk and heads."""
import math
import numpy as np
try:
    from .networks import QNetwork
    from .replay_buffer import ReplayBuffer
    from .normalizer import StateNormalizer
except ImportError:  # Legacy checkout imports.
    from networks import QNetwork
    from replay_buffer import ReplayBuffer
    from normalizer import StateNormalizer


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3,
                 hidden_dim=128, n_layers=3, eps_start=1.0, eps_end=0.05,
                 eps_decay=5000, target_update_freq=200, batch_size=64,
                 buffer_capacity=50_000, seed=42, double=False):
        self.name        = "DDQN" if double else "DQN"
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update_freq = target_update_freq
        self.batch_size  = batch_size
        self.double      = double
        self.steps_done  = 0
        self.update_count = 0
        self.rng         = np.random.default_rng(seed)

        self.online = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed)
        self.target = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed)
        self.target.copy_weights_from(self.online)

        self.buffer     = ReplayBuffer(buffer_capacity, seed=seed)
        self.normalizer = StateNormalizer(state_dim)
        self.last_loss  = 0.0

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay)

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        s = self.normalizer.normalize(state)[None]
        return int(self.online.forward(s).argmax())

    def store(self, state, action, reward, next_state, consequence, done):
        self.normalizer.update(state)
        self.buffer.push(state, action, reward, next_state, consequence, done)

    def update(self):
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return False

        S  = np.array([self.normalizer.normalize(s) for s in batch["states"]])
        NS = np.array([self.normalizer.normalize(s) for s in batch["next_states"]])
        A  = batch["actions"]
        R  = batch["rewards"]
        D  = batch["dones"]
        W  = batch["weights"]
        B  = len(A)

        next_q_target = self.target.forward(NS)
        next_acts     = (self.online.forward(NS) if self.double else next_q_target).argmax(axis=-1)
        next_q        = next_q_target[np.arange(B), next_acts]
        td_target     = R + self.gamma * next_q * (1.0 - D)

        q_all    = self.online.forward(S)
        q_cur    = q_all[np.arange(B), A]
        td_error = td_target - q_cur
        self.last_loss = float(np.mean(W * td_error**2))

        # Full backprop
        self.online.backward_update(S, A, td_error, W)
        self.buffer.update_priorities(batch["indices"], np.abs(td_error))

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target.copy_weights_from(self.online)
        return True

    def diagnostics(self):
        return {"epsilon": round(self.epsilon, 4), "steps": self.steps_done,
                "updates": self.update_count, "loss": round(self.last_loss, 4)}
