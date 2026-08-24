import numpy as np
try:
    from .networks import ActorNetwork, CriticNetwork
    from .normalizer import StateNormalizer
except ImportError:
    from networks import ActorNetwork, CriticNetwork
    from normalizer import StateNormalizer


class A2CAgent:
    name = "A2C"

    def __init__(self, state_dim, action_dim, gamma=0.99, lr_actor=3e-4,
                 lr_critic=1e-3, hidden_dim=128, n_layers=2,
                 entropy_coeff=0.01, n_steps=8, seed=42):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.entropy_coeff = entropy_coeff
        self.n_steps       = n_steps
        self.steps_done    = 0
        self.update_count  = 0
        self.rng           = np.random.default_rng(seed)

        self.actor      = ActorNetwork(state_dim, action_dim, hidden_dim, n_layers, lr_actor,  seed)
        self.critic     = CriticNetwork(state_dim,             hidden_dim, n_layers, lr_critic, seed)
        self.normalizer = StateNormalizer(state_dim)

        self._rollout: list = []
        self.last_actor_loss  = 0.0
        self.last_critic_loss = 0.0

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1
        s = self.normalizer.normalize(state)
        if eval_mode:
            return int(self.actor.logits(s[None]).squeeze().argmax())
        return self.actor.sample(s)

    def store(self, state, action, reward, next_state, consequence, done):
        s_norm  = self.normalizer.normalize(state)
        self.normalizer.update(state)
        ns_norm = self.normalizer.normalize(next_state)
        if len(self._rollout) >= self.n_steps * 2:
            self._rollout = self._rollout[-self.n_steps:]
        self._rollout.append(
            (s_norm, action, float(reward), ns_norm, bool(done)))

    def update(self):
        terminal_tail = bool(self._rollout and self._rollout[-1][-1])
        if len(self._rollout) < self.n_steps and not terminal_tail:
            return False

        T = min(len(self._rollout), self.n_steps)
        traj = self._rollout[:T]
        states, actions, rewards, next_states, dones = zip(*traj)

        S  = np.asarray(states, np.float32)
        NS = np.asarray(next_states, np.float32)
        A  = np.array(actions, np.int32)
        R  = np.array(rewards, np.float32)
        D  = np.array(dones,   np.float32)
        W  = np.ones(len(A),   np.float32)

        returns = np.zeros(T, np.float32)
        G = self.critic.value(NS[-1:]).item() * (1.0 - D[-1])
        for t in reversed(range(T)):
            G = R[t] + self.gamma * G * (1.0 - D[t])
            returns[t] = G

        values     = self.critic.value(S)
        advantages = np.clip(returns - values, -5.0, 5.0)

        self.actor.backward_update(
            S, A, advantages, W, entropy_coeff=self.entropy_coeff)
        v_errors = np.clip(returns - self.critic.value(S), -5.0, 5.0)
        self.critic.backward_update(S, v_errors, W)

        self.last_actor_loss  = float(np.mean((returns - values)**2))
        self.last_critic_loss = float(np.mean(v_errors**2))
        self.update_count    += 1
        del self._rollout[:T]
        return True

    def diagnostics(self):
        return {"steps": self.steps_done, "updates": self.update_count,
                "actor_loss":  round(self.last_actor_loss,  4),
                "critic_loss": round(self.last_critic_loss, 4)}
