import numpy as np


class BaseEnv:
    STATE_LABELS  = ("resource_load", "future_risk", "action_priority",
                     "system_pressure", "uncertainty", "hidden_penalty")
    ACTION_LABELS = ("DEFER", "PARTIAL", "FULL", "INVEST", "REBALANCE")
    state_dim  = 6
    action_dim = 5
    constraint_threshold = 3.0
    scm_labels_valid = False

    def __init__(self, max_steps=100, consequence_delay=5, noise_std=0.05,
                 reward_scale=1.0, penalty_shift=0.0, seed=0):
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if int(consequence_delay) < 0:
            raise ValueError("consequence_delay must be non-negative")
        if not np.isfinite(noise_std) or float(noise_std) < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        if not np.isfinite(reward_scale) or not np.isfinite(penalty_shift):
            raise ValueError("reward_scale and penalty_shift must be finite")
        self.max_steps         = int(max_steps)
        self.consequence_delay = int(consequence_delay)
        self.noise_std         = float(noise_std)
        self.reward_scale      = float(reward_scale)
        self.penalty_shift     = float(penalty_shift)
        self.rng               = np.random.default_rng(seed)
        self._state            = np.zeros(self.state_dim, np.float32)
        self._step             = 0
        self._done             = False
        self._total_reward = self._total_consequence = 0.0
        self._delayed_hits     = 0
        self._consequence_queue: list = []

    def reset(self):
        self._state             = self.rng.uniform(0.1, 0.6, self.state_dim).astype(np.float32)
        self._step              = 0
        self._done              = False
        self._total_reward      = self._total_consequence = 0.0
        self._delayed_hits      = 0
        self._consequence_queue = []
        self._last_actual_tau   = None
        return self._state.copy()

    def _transition(self, action):
        s = self._state.copy()
        rl, fr, ap, sp, unc, hpl = s

        if   action == 0: reward, drl, dfr, dsp, dhpl = 0.10*ap,             -0.05,  0.08*(1-ap),  0.05,  0.02
        elif action == 1: reward, drl, dfr, dsp, dhpl = 0.30*ap+0.10*(1-rl),  0.10, -0.04,        -0.03, -0.01
        elif action == 2: reward, drl, dfr, dsp, dhpl = 0.70*ap+0.20*(1-rl),  0.25, -0.10,        -0.06,  0.15*fr
        elif action == 3: reward, drl, dfr, dsp, dhpl = -0.15,                0.12, -0.20,         -0.08, -0.12
        else:             reward, drl, dfr, dsp, dhpl = 0.05,                -0.10,  0.02,         -0.15, -0.05

        reward *= self.reward_scale
        noise   = self.rng.normal(0, self.noise_std, self.state_dim).astype(np.float32)
        ns = np.array([
            np.clip(rl  + drl  + noise[0], 0, 1),
            np.clip(fr  + dfr  + noise[1], 0, 1),
            np.clip(self.rng.uniform(0.1, 0.9), 0, 1),
            np.clip(sp  + dsp  + noise[3], 0, 1),
            np.clip(unc + self.rng.normal(0, 0.03), 0, 1),
            np.clip(hpl + dhpl + noise[5] + self.penalty_shift*0.01, 0, 1),
        ], np.float32)

        F = np.clip(drl*fr + (hpl+self.penalty_shift)*float(action==2)*0.5, 0, 1)
        U = np.clip(max(0, ns[0]-0.6)/0.4 + drl*max(0, rl-0.5), 0, 1)
        D = np.clip(sp*float(action==2)*0.6 + unc*abs(dsp)*0.3, 0, 1)
        consequence = float(0.5*F + 0.3*U + 0.2*D)

        if self._step % 20 == 19:
            ns[1] = np.clip(ns[1] + self.rng.uniform(0.1, 0.3), 0, 1)
            ns[5] = np.clip(ns[5] + 0.15, 0, 1)

        info = {"F": float(F), "U": float(U), "D": float(D),
                "consequence": consequence, "resource_load": float(ns[0]),
                "future_risk": float(ns[1]), "system_pressure": float(ns[3])}
        return ns, reward, consequence, info

    def _get_delayed_consequence(self, immediate_c):
        self._consequence_queue.append(immediate_c)
        if len(self._consequence_queue) > self.consequence_delay:
            actual_tau = self.consequence_delay
            self._last_actual_tau = actual_tau
            return self._consequence_queue.pop(0)
        self._last_actual_tau = None
        return 0.0

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called after the episode terminated")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.action_dim:
            raise ValueError(f"action {action} outside [0, {self.action_dim})")
        ns, reward, imm_c, info = self._transition(action)

        self._step  += 1
        self._state  = ns
        self._done   = self._step >= self.max_steps or ns[0] >= 0.98

        delayed_c = self._get_delayed_consequence(imm_c)
        emitted_delayed_hit = (
            self._last_actual_tau is not None and delayed_c > 0.0)
        delay_supervision_valid = self._last_actual_tau is not None

        if self._done and self._consequence_queue:
            pending = self._consequence_queue
            delayed_c += float(np.sum(pending))
            self._delayed_hits += int(np.count_nonzero(np.asarray(pending) > 0.0))
            self._consequence_queue = []
            delay_supervision_valid = False

        if ns[0] >= 0.98:
            reward    -= 2.0 * self.reward_scale
            delayed_c += 0.5
            self._delayed_hits += 1
            delay_supervision_valid = False
            info["terminal_collapse_consequence"] = 0.5

        if emitted_delayed_hit:
            self._delayed_hits += 1

        self._total_reward      += reward
        self._total_consequence += delayed_c
        info.update({"step": self._step, "delayed_hits": self._delayed_hits,
                     "total_reward": self._total_reward,
                     "total_consequence": self._total_consequence,
                     "immediate_consequence": float(imm_c),
                     "actual_tau": (self._last_actual_tau
                                    if delay_supervision_valid else None),
                     "delay_supervision_valid": delay_supervision_valid,
                     "scm_label_valid": bool(self.scm_labels_valid)})
        return ns, float(reward), float(delayed_c), self._done, info

    def _record_step_adjustment(self, base_reward: float, base_cost: float,
                                reward: float, cost: float, info: dict,
                                delayed_event: bool | int = False):
        """Keep episode totals and returned ``info`` consistent in subclasses."""
        self._total_reward += float(reward) - float(base_reward)
        self._total_consequence += float(cost) - float(base_cost)
        if not np.isclose(cost, base_cost):
            info["actual_tau"] = None
            info["delay_supervision_valid"] = False
        if delayed_event and cost > base_cost:
            self._delayed_hits += int(delayed_event)
        info.update({
            "delayed_hits": self._delayed_hits,
            "total_reward": self._total_reward,
            "total_consequence": self._total_consequence,
        })

    @property
    def done(self): return self._done

    def episode_stats(self):
        return {"total_reward": self._total_reward,
                "total_consequence": self._total_consequence,
                "delayed_hits": self._delayed_hits, "steps": self._step}


class StandardEnv(BaseEnv):
    name = "standard"
    scm_labels_valid = True
    def __init__(self, **kw): super().__init__(noise_std=0.05, **kw)


class NoisyEnv(BaseEnv):
    name = "noisy"
    def __init__(self, **kw): super().__init__(noise_std=0.20, **kw)
    def step(self, action):
        ns, r, c, done, info = super().step(action)
        obs = np.clip(ns + self.rng.normal(0, 0.10, self.state_dim).astype(np.float32), 0, 1)
        return obs, r, c, done, info


class ShiftedConsequenceEnv(BaseEnv):
    name = "shifted"
    def __init__(self, **kw):
        kw.pop("consequence_delay", None)
        super().__init__(consequence_delay=10, noise_std=0.08, **kw)
    def _transition(self, action):
        ns, reward, consequence, info = super()._transition(action)
        if action == 2 and ns[1] > 0.5:
            consequence *= 1.8
        if (self._step + 1) % 15 == 0:
            consequence += float(ns[5] * 0.4)
        info["consequence"] = float(consequence)
        return ns, reward, consequence, info


class RandomisedEnv(BaseEnv):
    name = "randomised"

    def __init__(self, **kw):
        super().__init__(noise_std=0.10, **kw)
        self._shock_interval = 20

    def reset(self):
        self.noise_std         = self.rng.uniform(0.02, 0.25)
        self.reward_scale      = self.rng.uniform(0.6,  1.4)
        self.penalty_shift     = self.rng.uniform(-0.3, 0.3)
        self.consequence_delay = int(self.rng.integers(2, 12))
        self._shock_interval   = int(self.rng.integers(10, 30))
        return super().reset()

    def _transition(self, action):
        ns, reward, consequence, info = super()._transition(action)
        if (self._step + 1) % self._shock_interval == 0:
            consequence += float(ns[5] * self.rng.uniform(0.2, 0.6))
        info.update({
            "consequence": float(consequence),
            "resource_load": float(ns[0]),
            "future_risk": float(ns[1]),
        })
        return ns, reward, consequence, info


class AdversarialEnv(BaseEnv):
    """
    Adversarial: periodic adversary pushes resource_load and future_risk upward.
    Short-term greedy strategies collapse; consequence-aware policies survive.
    """
    name = "adversarial"

    def __init__(self, adversary_freq=8, adversary_strength=0.25, **kw):
        if int(adversary_freq) <= 0 or float(adversary_strength) < 0.0:
            raise ValueError("adversary_freq must be positive and strength non-negative")
        super().__init__(noise_std=0.07, **kw)
        self.adversary_freq     = int(adversary_freq)
        self.adversary_strength = float(adversary_strength)

    def _transition(self, action):
        ns, reward, consequence, info = super()._transition(action)
        if (self._step + 1) % self.adversary_freq == 0:
            ns[0] = np.clip(ns[0] + self.adversary_strength * self.rng.uniform(0.5, 1.0), 0, 1)
            ns[1] = np.clip(ns[1] + self.adversary_strength * self.rng.uniform(0.3, 0.8), 0, 1)
            consequence += float(ns[0] * self.adversary_strength * 0.5)
        info.update({
            "consequence": float(consequence),
            "resource_load": float(ns[0]),
            "future_risk": float(ns[1]),
        })
        return ns, reward, consequence, info


class DeceptiveRewardEnv(BaseEnv):
    """
    Deceptive reward: FULL (2) inflates immediate reward but accumulates a
    hidden penalty that bursts after a delay window. Tests causal reasoning.
    """
    name = "deceptive_reward"

    def __init__(self, deception_window=12, penalty_burst=2.0, **kw):
        if int(deception_window) <= 0 or float(penalty_burst) < 0.0:
            raise ValueError("deception_window must be positive and penalty_burst non-negative")
        kw.pop("consequence_delay", None)
        super().__init__(consequence_delay=int(deception_window), noise_std=0.06, **kw)
        self.deception_window = int(deception_window)
        self.penalty_burst    = float(penalty_burst)
        self._deception_acc   = 0.0

    def reset(self):
        self._deception_acc = 0.0
        return super().reset()

    def step(self, action):
        ns, r, c, done, info = super().step(action)
        base_r, base_c = r, c
        delayed_event = False
        if action == 2:
            r += 0.5 * self.reward_scale
            self._deception_acc += 0.3
        if self._step % self.deception_window == 0 and self._deception_acc > 0:
            c += min(self._deception_acc * self.penalty_burst, 3.0)
            self._deception_acc = 0.0
            delayed_event = True
        if done and self._deception_acc > 0:
            c += min(self._deception_acc * self.penalty_burst, 3.0)
            self._deception_acc = 0.0
            delayed_event = True
        self._record_step_adjustment(
            base_r, base_c, r, c, info, delayed_event=delayed_event)
        return ns, r, c, done, info


class ResourceCollapseEnv(BaseEnv):
    """
    Resource collapse: once resource_load exceeds a threshold it receives
    additional positive feedback. Recovery is costly; INVEST/REBALANCE can
    reduce the load.
    """
    name = "resource_collapse"

    def __init__(self, collapse_threshold=0.75, recovery_cost=3.0, **kw):
        if not 0.0 < float(collapse_threshold) < 1.0:
            raise ValueError("collapse_threshold must lie strictly between 0 and 1")
        if float(recovery_cost) < 0.0:
            raise ValueError("recovery_cost must be non-negative")
        super().__init__(noise_std=0.06, **kw)
        self.collapse_threshold = float(collapse_threshold)
        self.recovery_cost      = float(recovery_cost)
        self._in_collapse       = False

    def reset(self):
        self._in_collapse = False
        return super().reset()

    def _transition(self, action):
        ns, reward, consequence, info = super()._transition(action)

        if ns[0] >= self.collapse_threshold:
            self._in_collapse = True
            growth  = 0.05 * (ns[0] - self.collapse_threshold) / (1 - self.collapse_threshold + 1e-6)
            ns[0]   = np.clip(ns[0] + growth, 0, 1)
            consequence += growth * 2.0
        else:
            self._in_collapse = False

        if self._in_collapse and action in (3, 4):
            ns[0]   = np.clip(ns[0] - 0.12, 0, 1)
            reward += 0.3 * self.reward_scale
        elif self._in_collapse:
            consequence += self.recovery_cost * 0.05

        info.update({
            "consequence": float(consequence),
            "resource_load": float(ns[0]),
            "future_risk": float(ns[1]),
        })
        return ns, reward, consequence, info


ENV_REGISTRY = {
    "standard":          StandardEnv,
    "noisy":             NoisyEnv,
    "shifted":           ShiftedConsequenceEnv,
    "randomised":        RandomisedEnv,
    "adversarial":       AdversarialEnv,
    "deceptive_reward":  DeceptiveRewardEnv,
    "resource_collapse": ResourceCollapseEnv,
}

def make_env(name, **kw): return ENV_REGISTRY[name](**kw)
