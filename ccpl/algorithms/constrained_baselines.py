import math
import numpy as np
try:
    from .networks import ActorNetwork, CriticNetwork, QNetwork, softmax, sigmoid, Adam
    from .normalizer import StateNormalizer
    from .replay_buffer import ReplayBuffer
except ImportError:
    from networks import ActorNetwork, CriticNetwork, QNetwork, softmax, sigmoid, Adam
    from normalizer import StateNormalizer
    from replay_buffer import ReplayBuffer



class LagrangianMultiplier:
    """
    Learnable scalar Lagrange multiplier λ ≥ 0.
    Updated by gradient ascent on: λ * (J_c - d)
    where J_c is the mean episode cost and d is the constraint threshold.
    """
    def __init__(self, init_val: float = 1.0, lr: float = 5e-3,
                 lambda_max: float = 10.0):
        self._log_lam  = np.array([math.log(max(init_val, 1e-4))], np.float32)
        self.lr        = lr
        self.lambda_max = lambda_max
        self._optim    = Adam([self._log_lam], lr=lr)

    @property
    def value(self) -> float:
        return float(np.exp(np.clip(
            self._log_lam[0], -20.0, math.log(max(self.lambda_max, 1e-8)))))

    def update(self, mean_cost: float, threshold: float):
        """Gradient ascent: increase λ when cost > threshold."""
        grad = np.array([
            -self.value * (mean_cost - threshold)], np.float32)
        self._optim.step([grad])
        self._log_lam[0] = np.clip(
            self._log_lam[0], -20.0, math.log(max(self.lambda_max, 1e-8)))

    def __float__(self): return self.value



class CPOAgent:
    """
    First-order constrained-policy approximation.

    Key idea: perform a trust-region policy update (like TRPO) but project
    the update back into the feasible constraint set when necessary.

    Implementation (NumPy approximation):
      - On-policy rollout buffer (n_steps transitions)
      - GAE advantage estimation for reward and cost simultaneously
      - Policy update via gradient step clipped by KL proxy (δ_kl)
      - Cost constraint enforced by scaling back the update when
        the projected cost improvement exceeds the constraint budget

    This is not the conjugate-gradient/natural-gradient algorithm of Achiam et
    al. (2017), and benchmark labels must not present it as an exact CPO
    reproduction.  The class name is retained for backwards compatibility.
    """
    name = "CPO-FO"

    def __init__(self, state_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 cost_limit=3.0, lr_actor=2e-4, lr_critic=1e-3,
                 hidden_dim=128, n_layers=2, n_steps=64,
                 delta_kl=0.01, seed=42):
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.cost_limit  = cost_limit
        self.n_steps     = n_steps
        self.delta_kl    = delta_kl
        self.steps_done  = 0
        self.update_count = 0
        self.rng         = np.random.default_rng(seed)

        self.actor      = ActorNetwork(state_dim, action_dim, hidden_dim,
                                       n_layers, lr_actor, seed)
        self.critic     = CriticNetwork(state_dim, hidden_dim, n_layers,
                                        lr_critic, seed)
        self.cost_critic = CriticNetwork(state_dim, hidden_dim, n_layers,
                                         lr_critic, seed + 1)
        self.normalizer = StateNormalizer(state_dim)
        self._rollout: list = []
        self._ep_cost = 0.0
        self._ep_gamma = 1.0
        self._last_episode_cost = float(cost_limit)

        self.last_policy_loss = 0.0
        self.last_cost_return = 0.0

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1
        s = self.normalizer.normalize(state)
        if eval_mode:
            return int(self.actor.logits(s[None]).squeeze().argmax())
        return self.actor.sample(s)

    def store(self, state, action, reward, next_state, consequence, done):
        s  = self.normalizer.normalize(state)
        lp = float(np.log(
            np.clip(softmax(self.actor.logits(s[None]))[0, action], 1e-8, 1.0)
        ))
        self.normalizer.update(state)
        ns = self.normalizer.normalize(next_state)
        self._ep_cost += self._ep_gamma * float(consequence)
        self._ep_gamma *= self.gamma
        if done:
            self._last_episode_cost = self._ep_cost
            self._ep_cost, self._ep_gamma = 0.0, 1.0
        self._rollout.append((s, action, float(reward), float(consequence),
                               ns, bool(done), lp))

    def update(self):
        terminal_tail = bool(self._rollout and self._rollout[-1][5])
        if len(self._rollout) < self.n_steps and not terminal_tail:
            return False

        T = min(len(self._rollout), self.n_steps)
        traj = self._rollout[:T]
        states_r, actions, rewards, costs, next_states_r, dones, old_lps = zip(*traj)

        S   = np.asarray(states_r, np.float32)
        NS  = np.asarray(next_states_r, np.float32)
        A   = np.array(actions,  np.int32)
        R   = np.array(rewards,  np.float32)
        C   = np.array(costs,    np.float32)
        D   = np.array(dones,    np.float32)
        vals  = self.critic.value(S)
        nvals = self.critic.value(NS)
        adv_r = self._gae(R, vals, nvals, D)
        ret_r = adv_r + vals

        cvals  = self.cost_critic.value(S)
        cnvals = self.cost_critic.value(NS)
        adv_c  = self._gae(C, cvals, cnvals, D)
        ret_c  = adv_c + cvals

        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

        mean_cost = float(self._last_episode_cost)
        self.last_cost_return = mean_cost
        constraint_violated = mean_cost > self.cost_limit

        W_r = np.ones(T, np.float32)
        policy_adv = adv_r.copy()
        if constraint_violated:
            violation = mean_cost - self.cost_limit
            scale = float(np.clip(
                violation / (self.cost_limit + 1e-8), 0.0, 2.0))
            adv_c_n = adv_c / (adv_c.std() + 1e-8)
            policy_adv = policy_adv - scale * adv_c_n

        old_probs = self.actor.probs(S)
        params = self.actor.net.all_params()
        old_params = [p.copy() for p in params]
        self.actor.backward_update(S, A, policy_adv, W_r)
        new_probs = self.actor.probs(S)
        mean_kl = float(np.mean(np.sum(
            old_probs * (np.log(old_probs + 1e-8)
                         - np.log(new_probs + 1e-8)), axis=-1)))
        if mean_kl > self.delta_kl:
            trust_scale = float(np.sqrt(self.delta_kl / (mean_kl + 1e-8)))
            for p, old_p in zip(params, old_params):
                p[:] = old_p + trust_scale * (p - old_p)

        v_errs  = np.clip(ret_r - self.critic.value(S), -5.0, 5.0)
        cv_errs = np.clip(ret_c - self.cost_critic.value(S), -5.0, 5.0)
        self.critic.backward_update(S, v_errs, W_r)
        self.cost_critic.backward_update(S, cv_errs, W_r)

        self.last_policy_loss = float(np.mean((ret_r - self.critic.value(S))**2))
        self.update_count += 1
        del self._rollout[:T]
        return True

    def _gae(self, rewards, values, next_values, dones):
        T   = len(rewards)
        adv = np.zeros(T, np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    def diagnostics(self):
        return {
            "steps":        self.steps_done,
            "updates":      self.update_count,
            "policy_loss":  round(self.last_policy_loss, 4),
            "cost_return":  round(self.last_cost_return, 4),
            "cost_limit":   self.cost_limit,
        }



class RCPOAgent:
    """
    Reward-Constrained Policy Optimisation.

    Key idea: augment the reward with a Lagrangian penalty on the constraint,
    then update the multiplier by dual ascent. Unlike PPO-Lagrangian (which
    uses the same clipped surrogate), RCPO uses a discounted cost critic
    to propagate constraint signals over multiple timesteps.

    Reference: Tessler et al. "Reward Constrained Policy Optimization" (2018).
    """
    name = "RCPO"

    def __init__(self, state_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, cost_limit=3.0, lr_actor=3e-4, lr_critic=1e-3,
                 lr_lambda=1e-2, hidden_dim=128, n_layers=2, n_steps=64,
                 n_epochs=4, mini_batch_size=32, lambda_max=5.0, seed=42):
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.cost_limit      = cost_limit
        self.n_steps         = n_steps
        self.n_epochs        = n_epochs
        self.mini_batch_size = mini_batch_size
        self.steps_done      = 0
        self.update_count    = 0
        self.rng             = np.random.default_rng(seed)

        self.actor       = ActorNetwork(state_dim, action_dim, hidden_dim,
                                        n_layers, lr_actor, seed)
        self.critic      = CriticNetwork(state_dim, hidden_dim, n_layers,
                                         lr_critic, seed)
        self.cost_critic = CriticNetwork(state_dim, hidden_dim, n_layers,
                                         lr_critic, seed + 1)
        self.normalizer  = StateNormalizer(state_dim)
        self.lagrangian  = LagrangianMultiplier(init_val=1.0, lr=lr_lambda,
                                                lambda_max=lambda_max)
        self._rollout: list = []
        self._ep_cost = 0.0
        self._ep_gamma = 1.0
        self._last_episode_cost = float(cost_limit)

        self.last_policy_loss = 0.0
        self.last_cost_return = 0.0

    def _log_prob(self, states, actions):
        p = softmax(self.actor.logits(states))
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
        lp = float(self._log_prob(s[None], np.array([action])).item())
        self.normalizer.update(state)
        ns = self.normalizer.normalize(next_state)
        self._ep_cost += self._ep_gamma * float(consequence)
        self._ep_gamma *= self.gamma
        if done:
            self._last_episode_cost = self._ep_cost
            self.lagrangian.update(self._last_episode_cost, self.cost_limit)
            self.last_cost_return = self._last_episode_cost
            self._ep_cost, self._ep_gamma = 0.0, 1.0
        self._rollout.append((s, action, float(reward), float(consequence),
                               ns, bool(done), lp))

    def update(self):
        terminal_tail = bool(self._rollout and self._rollout[-1][5])
        if len(self._rollout) < self.n_steps and not terminal_tail:
            return False

        T = min(len(self._rollout), self.n_steps)
        traj = self._rollout[:T]
        states_r, actions, rewards, costs, next_states_r, dones, old_lps = zip(*traj)

        S   = np.asarray(states_r, np.float32)
        NS  = np.asarray(next_states_r, np.float32)
        A   = np.array(actions,  np.int32)
        R   = np.array(rewards,  np.float32)
        C   = np.array(costs,    np.float32)
        D   = np.array(dones,    np.float32)
        OLP = np.array(old_lps,  np.float32)
        lam = self.lagrangian.value

        R_aug = R - lam * C

        vals  = self.critic.value(S)
        nvals = self.critic.value(NS)
        deltas = R_aug + self.gamma * nvals * (1 - D) - vals
        adv    = np.zeros(T, np.float32)
        gae    = 0.0
        for t in reversed(range(T)):
            gae    = deltas[t] + self.gamma * self.gae_lambda * (1 - D[t]) * gae
            adv[t] = gae
        returns = adv + vals
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        cvals  = self.cost_critic.value(S)
        cnvals = self.cost_critic.value(NS)
        c_adv  = np.zeros(T, np.float32)
        gae_c  = 0.0
        for t in reversed(range(T)):
            d_c     = C[t] + self.gamma * cnvals[t] * (1 - D[t]) - cvals[t]
            gae_c   = d_c + self.gamma * self.gae_lambda * (1 - D[t]) * gae_c
            c_adv[t] = gae_c
        c_returns = c_adv + cvals

        p_losses = []
        for _ in range(self.n_epochs):
            idxs = self.rng.permutation(T)
            for start in range(0, T, self.mini_batch_size):
                mb  = idxs[start:start + self.mini_batch_size]
                s_mb = S[mb]; a_mb = A[mb]
                adv_mb = adv[mb]; ret_mb = returns[mb]; olp_mb = OLP[mb]
                W_mb   = np.ones(len(mb), np.float32)

                new_lp = self._log_prob(s_mb, a_mb)
                ratio  = np.exp(np.clip(new_lp - olp_mb, -2, 2))
                clip_r = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                p_loss = -np.minimum(ratio * adv_mb, clip_r * adv_mb).mean()
                p_losses.append(float(p_loss))

                self.actor.backward_ppo(
                    s_mb, a_mb, adv_mb, olp_mb, self.clip_eps, W_mb)
                v_errs = np.clip(ret_mb - self.critic.value(s_mb), -5.0, 5.0)
                self.critic.backward_update(s_mb, v_errs, W_mb)

        W_full = np.ones(T, np.float32)
        cv_errs = np.clip(c_returns - self.cost_critic.value(S), -5.0, 5.0)
        self.cost_critic.backward_update(S, cv_errs, W_full)

        self.last_policy_loss = float(np.mean(p_losses)) if p_losses else 0.0
        self.update_count    += 1
        del self._rollout[:T]
        return True

    def diagnostics(self):
        return {
            "steps":       self.steps_done,
            "updates":     self.update_count,
            "policy_loss": round(self.last_policy_loss, 4),
            "cost_return": round(self.last_cost_return, 4),
            "lambda":      round(self.lagrangian.value, 4),
            "cost_limit":  self.cost_limit,
        }



class PIDLagrangianAgent:
    """
    PID-Lagrangian.

    Replaces the simple gradient-ascent dual update with a PID controller
    that drives J_c → d. The integral term prevents oscillation; the
    derivative term provides damping when the cost is rapidly changing.

    Reference: Stooke et al. "Responsive Safety in Reinforcement Learning
    by PID Lagrangian Methods" (2020).
    """
    name = "PID-Lag"

    def __init__(self, state_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, cost_limit=3.0, lr_actor=3e-4, lr_critic=1e-3,
                 hidden_dim=128, n_layers=2, n_steps=64, n_epochs=4,
                 mini_batch_size=32,
                 pid_kp=0.1, pid_ki=0.01, pid_kd=0.005,
                 lambda_max=10.0, seed=42):
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.cost_limit      = cost_limit
        self.n_steps         = n_steps
        self.n_epochs        = n_epochs
        self.mini_batch_size = mini_batch_size
        self.steps_done      = 0
        self.update_count    = 0
        self.rng             = np.random.default_rng(seed)

        self.kp, self.ki, self.kd = pid_kp, pid_ki, pid_kd
        self.lambda_max  = lambda_max
        self._lambda     = 1.0
        self._integral   = 0.0
        self._prev_error = 0.0

        self.actor      = ActorNetwork(state_dim, action_dim, hidden_dim,
                                       n_layers, lr_actor, seed)
        self.critic     = CriticNetwork(state_dim, hidden_dim, n_layers,
                                        lr_critic, seed)
        self.normalizer = StateNormalizer(state_dim)
        self._rollout: list = []
        self._ep_cost = 0.0
        self._ep_gamma = 1.0
        self._last_episode_cost = float(cost_limit)

        self.last_policy_loss = 0.0
        self.last_cost_return = 0.0

    def _pid_update(self, mean_cost: float):
        error          = mean_cost - self.cost_limit
        self._integral = np.clip(self._integral + error, -10.0, 10.0)
        derivative     = error - self._prev_error
        self._prev_error = error
        delta = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._lambda = float(np.clip(self._lambda + delta, 0.0, self.lambda_max))

    def _log_prob(self, states, actions):
        p = softmax(self.actor.logits(states))
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
        lp = float(self._log_prob(s[None], np.array([action])).item())
        self.normalizer.update(state)
        ns = self.normalizer.normalize(next_state)
        self._ep_cost += self._ep_gamma * float(consequence)
        self._ep_gamma *= self.gamma
        if done:
            self._last_episode_cost = self._ep_cost
            self._pid_update(self._last_episode_cost)
            self.last_cost_return = self._last_episode_cost
            self._ep_cost, self._ep_gamma = 0.0, 1.0
        self._rollout.append((s, action, float(reward), float(consequence),
                               ns, bool(done), lp))

    def update(self):
        terminal_tail = bool(self._rollout and self._rollout[-1][5])
        if len(self._rollout) < self.n_steps and not terminal_tail:
            return False

        T = min(len(self._rollout), self.n_steps)
        traj = self._rollout[:T]
        states_r, actions, rewards, costs, next_states_r, dones, old_lps = zip(*traj)

        S   = np.asarray(states_r, np.float32)
        NS  = np.asarray(next_states_r, np.float32)
        A   = np.array(actions,  np.int32)
        R   = np.array(rewards,  np.float32)
        C   = np.array(costs,    np.float32)
        D   = np.array(dones,    np.float32)
        OLP = np.array(old_lps,  np.float32)

        lam = self._lambda

        R_aug = R - lam * C
        vals  = self.critic.value(S)
        nvals = self.critic.value(NS)
        deltas = R_aug + self.gamma * nvals * (1 - D) - vals
        adv    = np.zeros(T, np.float32)
        gae    = 0.0
        for t in reversed(range(T)):
            gae    = deltas[t] + self.gamma * self.gae_lambda * (1 - D[t]) * gae
            adv[t] = gae
        returns = adv + vals
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        p_losses = []
        for _ in range(self.n_epochs):
            idxs = self.rng.permutation(T)
            for start in range(0, T, self.mini_batch_size):
                mb     = idxs[start:start + self.mini_batch_size]
                s_mb   = S[mb]; a_mb = A[mb]
                adv_mb = adv[mb]; ret_mb = returns[mb]; olp_mb = OLP[mb]
                W_mb   = np.ones(len(mb), np.float32)

                new_lp = self._log_prob(s_mb, a_mb)
                ratio  = np.exp(np.clip(new_lp - olp_mb, -2, 2))
                clip_r = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                p_loss = -np.minimum(ratio * adv_mb, clip_r * adv_mb).mean()
                p_losses.append(float(p_loss))

                self.actor.backward_ppo(
                    s_mb, a_mb, adv_mb, olp_mb, self.clip_eps, W_mb)
                v_errs = np.clip(ret_mb - self.critic.value(s_mb), -5.0, 5.0)
                self.critic.backward_update(s_mb, v_errs, W_mb)

        self.last_policy_loss = float(np.mean(p_losses)) if p_losses else 0.0
        self.update_count    += 1
        del self._rollout[:T]
        return True

    def diagnostics(self):
        return {
            "steps":       self.steps_done,
            "updates":     self.update_count,
            "policy_loss": round(self.last_policy_loss, 4),
            "cost_return": round(self.last_cost_return, 4),
            "lambda":      round(self._lambda, 4),
            "integral":    round(self._integral, 4),
            "cost_limit":  self.cost_limit,
        }



class SACLagrangianAgent:
    """
    SAC-Lagrangian (off-policy).

    Soft Actor-Critic with entropy regularisation and a Lagrangian multiplier
    on the constraint cost. The policy maximises:
        J(π) = E[r - λ·c + α·H(π)]

    Uses twin Q-networks (like SAC) to reduce overestimation, plus a separate
    cost Q-network. Lagrangian multiplier updated by gradient ascent.

    Reference: Ha et al. "Learning to Walk in Minutes Using Massively
    Parallel Deep Reinforcement Learning" (adapted with cost critic).
    """
    name = "SAC-Lag"

    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005,
                 lr=3e-4, lr_alpha=3e-4, lr_lambda=5e-3,
                 hidden_dim=128, n_layers=2,
                 batch_size=64, buffer_capacity=50_000,
                 cost_limit=3.0, lambda_max=5.0,
                 target_entropy_ratio=0.5, seed=42):
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.cost_limit  = cost_limit
        self.steps_done  = 0
        self.update_count = 0
        self.rng         = np.random.default_rng(seed)

        self.q1        = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed)
        self.q2        = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed+1)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed+1)
        self.q1_target.copy_weights_from(self.q1)
        self.q2_target.copy_weights_from(self.q2)

        self.q_cost        = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed+2)
        self.q_cost_target = QNetwork(state_dim, action_dim, hidden_dim, n_layers, lr, seed+2)
        self.q_cost_target.copy_weights_from(self.q_cost)

        self.actor      = ActorNetwork(state_dim, action_dim, hidden_dim,
                                       n_layers, lr, seed)
        self.normalizer = StateNormalizer(state_dim)
        self.buffer     = ReplayBuffer(buffer_capacity, seed=seed)

        self.target_entropy = -target_entropy_ratio * np.log(1.0 / action_dim)
        self._log_alpha = np.array([0.0], np.float32)
        self._alpha_optim = Adam([self._log_alpha], lr=lr_alpha)

        self.lagrangian = LagrangianMultiplier(init_val=1.0, lr=lr_lambda,
                                               lambda_max=lambda_max)

        self.last_policy_loss = 0.0
        self.last_cost_return = 0.0
        self._ep_cost = 0.0
        self._ep_gamma = 1.0
        self._last_episode_cost = float(cost_limit)

    @property
    def alpha(self) -> float:
        return float(np.exp(np.clip(self._log_alpha[0], -20.0, 5.0)))

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1
        s = self.normalizer.normalize(state)[None]
        if eval_mode:
            return int(self.actor.logits(s).squeeze().argmax())
        return self.actor.sample(self.normalizer.normalize(state))

    def store(self, state, action, reward, next_state, consequence, done):
        self.normalizer.update(state)
        self.buffer.push(state, action, reward, next_state, consequence, done)
        self._ep_cost += self._ep_gamma * float(consequence)
        self._ep_gamma *= self.gamma
        if done:
            self._last_episode_cost = self._ep_cost
            self.lagrangian.update(self._last_episode_cost, self.cost_limit)
            self.last_cost_return = self._last_episode_cost
            self._ep_cost, self._ep_gamma = 0.0, 1.0

    def update(self):
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return False

        S  = np.array([self.normalizer.normalize(s) for s in batch["states"]])
        NS = np.array([self.normalizer.normalize(s) for s in batch["next_states"]])
        A  = batch["actions"]
        R  = batch["rewards"]
        C  = batch["consequences"]
        D  = batch["dones"]
        W  = batch["weights"]
        B  = len(A)
        lam = self.lagrangian.value

        next_probs = softmax(self.actor.logits(NS))
        next_log_p = np.log(next_probs + 1e-8)

        q1_ns = self.q1_target.forward(NS)
        q2_ns = self.q2_target.forward(NS)
        q_ns  = np.minimum(q1_ns, q2_ns)
        v_ns  = (next_probs * (q_ns - self.alpha * next_log_p)).sum(-1)
        td_r  = R + self.gamma * v_ns * (1 - D)

        qc_ns  = self.q_cost_target.forward(NS)
        vc_ns  = (next_probs * qc_ns).sum(-1)
        td_c   = C + self.gamma * vc_ns * (1 - D)

        q1_cur = self.q1.forward(S)[np.arange(B), A]
        q2_cur = self.q2.forward(S)[np.arange(B), A]
        qc_cur = self.q_cost.forward(S)[np.arange(B), A]

        td_err1 = td_r - q1_cur
        td_err2 = td_r - q2_cur
        td_errc = td_c - qc_cur

        self.q1.backward_update(S, A, td_err1, W)
        self.q2.backward_update(S, A, td_err2, W)
        self.q_cost.backward_update(S, A, td_errc, W)

        probs   = softmax(self.actor.logits(S))
        log_p   = np.log(probs + 1e-8)
        q1_s    = self.q1.forward(S)
        q2_s    = self.q2.forward(S)
        q_s     = np.minimum(q1_s, q2_s)
        qc_s    = self.q_cost.forward(S)

        self.actor.backward_discrete_objective(
            S, q_s, qc_s, lam, self.alpha, W)
        policy_loss = float(np.mean(np.sum(
            probs * (self.alpha * log_p - q_s + lam * qc_s), axis=-1)))

        entropy = -(probs * log_p).sum(-1).mean()
        alpha_grad = np.array([
            self.alpha * (entropy - self.target_entropy)], np.float32)
        self._alpha_optim.step([alpha_grad])
        self._log_alpha[0] = np.clip(self._log_alpha[0], -20.0, 5.0)

        self.q1_target.soft_update_from(self.q1, self.tau)
        self.q2_target.soft_update_from(self.q2, self.tau)
        self.q_cost_target.soft_update_from(self.q_cost, self.tau)

        self.buffer.update_priorities(
            batch["indices"],
            0.5 * (np.abs(td_err1) + np.abs(td_err2)),
            c_errors=np.abs(td_errc))

        self.last_policy_loss = policy_loss
        self.update_count    += 1
        return True

    def diagnostics(self):
        return {
            "steps":       self.steps_done,
            "updates":     self.update_count,
            "policy_loss": round(self.last_policy_loss, 4),
            "cost_return": round(self.last_cost_return, 4),
            "lambda":      round(self.lagrangian.value, 4),
            "alpha":       round(self.alpha, 4),
            "cost_limit":  self.cost_limit,
        }
