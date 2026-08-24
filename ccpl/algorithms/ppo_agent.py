"""PPO Agent — Proximal Policy Optimization with GAE and full backprop."""
import numpy as np
try:
    from .networks import ActorNetwork, CriticNetwork, softmax
    from .normalizer import StateNormalizer
except ImportError:
    from networks import ActorNetwork, CriticNetwork, softmax
    from normalizer import StateNormalizer


class PPOAgent:
    name = "PPO"

    def __init__(self, state_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, lr_actor=3e-4, lr_critic=1e-3, hidden_dim=128,
                 n_layers=2, entropy_coeff=0.01, n_steps=64, n_epochs=4,
                 mini_batch_size=32, seed=42):
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.entropy_coeff   = entropy_coeff
        self.n_steps         = n_steps
        self.n_epochs        = n_epochs
        self.mini_batch_size = mini_batch_size
        self.steps_done      = 0
        self.update_count    = 0
        self.rng             = np.random.default_rng(seed)

        self.actor      = ActorNetwork(state_dim, action_dim, hidden_dim, n_layers, lr_actor,  seed)
        self.critic     = CriticNetwork(state_dim,             hidden_dim, n_layers, lr_critic, seed)
        self.normalizer = StateNormalizer(state_dim)

        self._rollout: list = []
        self.last_policy_loss = 0.0
        self.last_value_loss  = 0.0

    def _log_prob(self, states, actions):
        p = self.actor.probs(states)
        return np.log(p[np.arange(len(actions)), actions] + 1e-8)

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1
        s = self.normalizer.normalize(state)
        if eval_mode:
            return int(self.actor.logits(s[None]).squeeze().argmax())
        return self.actor.sample(s)

    def store(self, state, action, reward, next_state, consequence, done):
        s  = self.normalizer.normalize(state)
        lp = self._log_prob(s[None], np.array([action])).item()
        self.normalizer.update(state)
        ns = self.normalizer.normalize(next_state)
        self._rollout.append(
            (s, action, float(reward), ns, bool(done), lp))

    def update(self):
        terminal_tail = bool(self._rollout and self._rollout[-1][4])
        if len(self._rollout) < self.n_steps and not terminal_tail:
            return False

        T = min(len(self._rollout), self.n_steps)
        traj = self._rollout[:T]
        states_r, actions, rewards, next_states_r, dones, old_lps = zip(*traj)

        S   = np.asarray(states_r, np.float32)
        NS  = np.asarray(next_states_r, np.float32)
        A   = np.array(actions, np.int32)
        R   = np.array(rewards, np.float32)
        D   = np.array(dones,   np.float32)
        OLP = np.array(old_lps, np.float32)

        vals   = self.critic.value(S)
        nvals  = self.critic.value(NS)
        deltas = R + self.gamma * nvals * (1 - D) - vals
        adv    = np.zeros(T, np.float32)
        gae    = 0.0
        for t in reversed(range(T)):
            gae    = deltas[t] + self.gamma * self.gae_lambda * (1 - D[t]) * gae
            adv[t] = gae
        returns = adv + vals
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        p_losses, v_losses = [], []

        for _ in range(self.n_epochs):
            idxs = self.rng.permutation(T)
            for start in range(0, T, self.mini_batch_size):
                mb      = idxs[start:start + self.mini_batch_size]
                if len(mb) == 0:
                    continue
                s_mb    = S[mb]; a_mb = A[mb]
                adv_mb  = adv[mb]; ret_mb = returns[mb]; olp_mb = OLP[mb]
                W_mb    = np.ones(len(mb), np.float32)

                if (not np.all(np.isfinite(adv_mb))
                        or not np.all(np.isfinite(ret_mb))):
                    continue

                new_lp  = self._log_prob(s_mb, a_mb)
                ratio   = np.exp(np.clip(new_lp - olp_mb, -2, 2))
                clip_r  = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                p_loss  = -np.minimum(ratio * adv_mb, clip_r * adv_mb).mean()

                vals_mb = self.critic.value(s_mb)
                v_loss  = np.mean((ret_mb - vals_mb)**2)

                self.actor.backward_ppo(
                    s_mb, a_mb, adv_mb, olp_mb, self.clip_eps, W_mb,
                    entropy_coeff=self.entropy_coeff)
                v_errs = np.clip(ret_mb - self.critic.value(s_mb), -5.0, 5.0)
                self.critic.backward_update(s_mb, v_errs, W_mb)

                p_losses.append(float(p_loss))
                v_losses.append(float(v_loss))

        self.last_policy_loss = float(np.mean(p_losses)) if p_losses else 0.0
        self.last_value_loss  = float(np.mean(v_losses)) if v_losses else 0.0
        self.update_count    += 1
        del self._rollout[:T]
        return True

    def diagnostics(self):
        return {"steps": self.steps_done, "updates": self.update_count,
                "policy_loss": round(self.last_policy_loss, 4),
                "value_loss":  round(self.last_value_loss,  4)}
