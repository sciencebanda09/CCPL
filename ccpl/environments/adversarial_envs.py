import numpy as np

try:
    from .environments import BaseEnv, ENV_REGISTRY
except ImportError:  # Legacy checkout imports.
    from environments import BaseEnv, ENV_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DeceptionBench  —  THE PPO collapse environment
# ─────────────────────────────────────────────────────────────────────────────

class DeceptionBenchEnv(BaseEnv):
    name = "deception_bench"
    constraint_threshold = 5.0   # budget d — see note above (FIX-B14)

    def __init__(self, counter_threshold: int = 5,
                 burst_magnitude: float = 12.0,
                 penalty_delay: int = 8,
                 warmup_steps: int = 20,
                 burst_queue_max: int = 10,   # FIX-B13: max queued bursts
                 **kw):
        if int(counter_threshold) <= 0 or int(penalty_delay) < 0:
            raise ValueError("counter_threshold must be positive and penalty_delay non-negative")
        if float(burst_magnitude) < 0.0 or int(warmup_steps) < 0:
            raise ValueError("burst_magnitude and warmup_steps must be non-negative")
        kw.pop("consequence_delay", None)
        super().__init__(consequence_delay=penalty_delay,
                         noise_std=0.05, **kw)
        self.counter_threshold = counter_threshold
        self.burst_magnitude   = burst_magnitude
        self.penalty_delay     = penalty_delay
        self.warmup_steps      = warmup_steps
        self.burst_queue_max   = burst_queue_max   # FIX-B13
        self._full_counter     = 0
        self._burst_queue: list = []

    def reset(self):
        self._full_counter = 0
        self._burst_queue  = []
        return super().reset()

    def step(self, action):
        ns, r, c, done, info = super().step(action)
        base_r, base_c = r, c
        bursts_fired = 0

        # Deceptive reward: FULL earns bonus during warmup-cleared phase
        if action == 2 and self._step > self.warmup_steps:
            r += 2.0

        # Counter: accumulates FULL actions post-warmup
        if action == 2 and self._step > self.warmup_steps:
            self._full_counter += 1

        # When counter fires, queue a burst consequence
        if self._full_counter >= self.counter_threshold:
            fire_at = self._step + self.penalty_delay

            # Never discard a caused consequence.  The old queue cap silently
            # removed the oldest pending burst without transferring its mass.
            self._burst_queue.append(fire_at)
            self._full_counter = 0          # reset; but damage queued

        # Apply matured bursts
        while self._burst_queue and self._step >= self._burst_queue[0]:
            c += self.burst_magnitude
            r -= self.burst_magnitude * 0.5  # partial reward claw-back
            bursts_fired += 1
            self._burst_queue.pop(0)

        if done and self._burst_queue:
            # A time limit must not erase consequences already caused inside
            # the episode.  Flush the tail and invalidate a single-delay label.
            pending = len(self._burst_queue)
            c += pending * self.burst_magnitude
            r -= pending * self.burst_magnitude * 0.5
            bursts_fired += pending
            self._burst_queue.clear()

        self._record_step_adjustment(
            base_r, base_c, r, c, info, delayed_event=bursts_fired)
        return ns, r, c, done, info

    def episode_stats(self):
        s = super().episode_stats()
        s["csr_violation"] = int(s["total_consequence"] > self.constraint_threshold)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HiddenStateShift  —  GRU ablation environment
# ─────────────────────────────────────────────────────────────────────────────

class HiddenStateShiftEnv(BaseEnv):
    name = "hidden_state_shift"
    constraint_threshold = 6.0

    def __init__(self, shift_step: int = 30,
                 post_multiplier: float = 3.5,
                 **kw):
        if int(shift_step) <= 0 or float(post_multiplier) < 0.0:
            raise ValueError("shift_step must be positive and post_multiplier non-negative")
        super().__init__(noise_std=0.06, **kw)
        self.shift_step      = shift_step
        self.post_multiplier = post_multiplier
        self._shifted        = False

    def reset(self):
        self._shifted = False
        return super().reset()

    def _transition(self, action):
        ns, reward, consequence, info = super()._transition(action)
        # Activate shift silently at shift_step
        if self._step + 1 >= self.shift_step and not self._shifted:
            self._shifted = True

        # Modify the source consequence before it enters the delay queue.
        if self._shifted and action == 2:
            consequence *= self.post_multiplier

        info["shifted"] = self._shifted
        info["consequence"] = float(consequence)
        return ns, reward, consequence, info

    def episode_stats(self):
        s = super().episode_stats()
        s["csr_violation"] = int(s["total_consequence"] > self.constraint_threshold)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ConflictZone  —  Constrained RL stress test
# ─────────────────────────────────────────────────────────────────────────────

class ConflictZoneEnv(BaseEnv):
    name = "conflict_zone"
    constraint_threshold = 8.0

    def __init__(self, conflict_fraction: float = 0.40,
                 step_limit: float = 0.5,
                 **kw):
        if not 0.0 <= float(conflict_fraction) <= 1.0 or float(step_limit) < 0.0:
            raise ValueError("conflict_fraction must be in [0,1] and step_limit non-negative")
        super().__init__(noise_std=0.07, **kw)
        self.conflict_fraction = conflict_fraction
        self.step_limit        = step_limit   # per-step constraint budget

    def reset(self):
        return super().reset()

    def _transition(self, action):
        ns, reward, consequence, info = super()._transition(action)
        # Determine if this step is a conflict step
        in_conflict = self.rng.random() < self.conflict_fraction
        info["in_conflict"] = in_conflict

        if in_conflict and action == 2:
            # FULL in conflict: reward is high, but constraint is violated
            reward *= 1.5
            consequence += self.step_limit + 0.3
        elif in_conflict:
            # Safe actions in conflict zone: slight penalty
            reward -= 0.1

        info["consequence"] = float(consequence)
        return ns, reward, consequence, info

    def episode_stats(self):
        s = super().episode_stats()
        s["csr_violation"] = int(s["total_consequence"] > self.constraint_threshold)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Safety Gymnasium adapter
# ─────────────────────────────────────────────────────────────────────────────

class SafetyGymAdapter:
    state_dim  = None  # set after __init__
    action_dim = None
    constraint_threshold = 25.0   # configurable evaluation budget

    def __init__(self, task_name: str,
                 constraint_threshold: float = 25.0,
                 action_dim: int | None = None,
                 action_wrapper: str = "axis",
                 max_steps: int = 1000,
                 seed: int = 0,
                 consequence_delay: int = 0,   # accepted but ignored (Safety Gym has own delay semantics)
                 **_kw):
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if not np.isfinite(constraint_threshold) or constraint_threshold < 0:
            raise ValueError("constraint_threshold must be finite and non-negative")
        if action_wrapper != "axis":
            raise ValueError(
                "Only the discrete neutral-plus-signed-axis wrapper is supported")
        try:
            import safety_gymnasium
            self._raw = safety_gymnasium.make(task_name, max_episode_steps=max_steps)
        except ImportError:
            raise ImportError(
                "safety-gymnasium not installed.  Run: pip install safety-gymnasium"
            )
        self.name = task_name
        self.constraint_threshold = constraint_threshold
        self.max_steps    = max_steps
        self.consequence_delay = 0
        self.action_wrapper = action_wrapper
        self.state_dim    = int(self._raw.observation_space.shape[0])  # ensure int not np.int
        raw_shape = getattr(self._raw.action_space, "shape", None)
        if not raw_shape:
            raise TypeError("SafetyGymAdapter requires a continuous Box action space")
        self.raw_action_dim = int(np.prod(raw_shape))
        # Neutral plus positive/negative movement on each actuator.  The old
        # 5-bin interpolation drove every actuator to the same value and could
        # not express most controls (especially SafetyAnt tasks).
        self.action_dim = 2 * self.raw_action_dim + 1
        self.requested_action_dim = action_dim  # retained for config compatibility
        self._seed        = int(seed)   # store for per-episode reproducibility

        self._step        = 0
        self._done        = False
        self._cum_cost    = 0.0
        self._cum_reward  = 0.0
        self._delayed_hits = 0

    # ── Environment interface ──────────────────────────────────────────────

    def reset(self, seed: int | None = None):
        # Forward seed to underlying env — required for reproducibility across seeds.
        # Older Safety Gym versions don't accept seed kwarg; fall back gracefully.
        _seed = seed if seed is not None else self._seed
        try:
            reset_out = self._raw.reset(seed=_seed)
        except TypeError:
            reset_out = self._raw.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self._step        = 0
        self._done        = False
        self._cum_cost    = 0.0
        self._cum_reward  = 0.0
        self._delayed_hits = 0
        return np.asarray(obs, np.float32)

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called after the episode terminated")
        raw_action = self._map_action(action)
        outcome = self._raw.step(raw_action)
        if len(outcome) == 6:
            obs, reward, cost, terminated, truncated, info = outcome
        elif len(outcome) == 5:  # compatibility with older wrappers
            obs, reward, terminated, truncated, info = outcome
            cost = info.get("cost", 0.0)
        else:
            raise RuntimeError(
                f"Unexpected Safety Gymnasium step tuple of length {len(outcome)}"
            )
        cost = float(cost)
        info = dict(info)
        info.setdefault("cost", cost)
        info.setdefault("actual_tau", 0)
        info.setdefault("delay_supervision_valid", True)
        info.setdefault("scm_label_valid", False)

        self._step        += 1
        self._cum_cost    += cost
        self._cum_reward  += reward
        self._done = terminated or truncated or self._step >= self.max_steps

        if cost > 0:
            self._delayed_hits += 1

        info.update({
            "step": self._step,
            "delayed_hits": self._delayed_hits,
            "total_reward": self._cum_reward,
            "total_consequence": self._cum_cost,
            "immediate_consequence": cost,
        })

        return (np.asarray(obs, np.float32),
                float(reward), cost, self._done, info)

    @property
    def done(self): return self._done

    def episode_stats(self):
        return {
            "total_reward":      self._cum_reward,
            "total_consequence": self._cum_cost,
            "delayed_hits":      self._delayed_hits,
            "steps":             self._step,
            "csr_violation":     int(self._cum_cost > self.constraint_threshold),
        }

    # ── Action mapping ─────────────────────────────────────────────────────

    def _map_action(self, discrete_action: int):
        """Map a discrete action to neutral or one signed actuator axis."""
        discrete_action = int(discrete_action)
        if not 0 <= discrete_action < self.action_dim:
            raise ValueError(
                f"action {discrete_action} outside [0, {self.action_dim})"
            )
        act_space = self._raw.action_space
        lo = np.asarray(act_space.low, np.float32).reshape(-1)
        hi = np.asarray(act_space.high, np.float32).reshape(-1)
        action = ((lo + hi) * 0.5).astype(np.float32)
        if discrete_action:
            axis = (discrete_action - 1) // 2
            positive = (discrete_action - 1) % 2 == 0
            action[axis] = hi[axis] if positive else lo[axis]
        return action.reshape(act_space.shape)

    def close(self):
        return self._raw.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CMDP wrapper for existing custom environments
# ─────────────────────────────────────────────────────────────────────────────

class CMDPWrapper:
    def __init__(self, base_env: BaseEnv, threshold: float = 8.0):
        self._env = base_env
        self.constraint_threshold = threshold
        self.name      = getattr(base_env, "name", "wrapped")
        self.state_dim = getattr(base_env, "state_dim", 6)
        self.action_dim = getattr(base_env, "action_dim", 5)

    def reset(self):           return self._env.reset()
    def step(self, a):         return self._env.step(a)
    @property
    def done(self):            return self._env.done

    def episode_stats(self):
        s = self._env.episode_stats()
        s["csr_violation"] = int(s["total_consequence"] > self.constraint_threshold)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

ADVERSARIAL_ENV_REGISTRY = {
    "deception_bench":    DeceptionBenchEnv,
    "hidden_state_shift": HiddenStateShiftEnv,
    "conflict_zone":      ConflictZoneEnv,
}

# Merge into the main ENV_REGISTRY so train.py / evaluate.py pick them up
ENV_REGISTRY.update(ADVERSARIAL_ENV_REGISTRY)

ADVERSARIAL_ENVS = tuple(ADVERSARIAL_ENV_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Safety Gymnasium task registry
#     Registered only when safety-gymnasium is installed; silently skipped
#     otherwise so the rest of the codebase works without it.
# ─────────────────────────────────────────────────────────────────────────────

def _make_safety_factory(task_name: str,
                          constraint_threshold: float = 25.0,
                          action_dim: int | None = None,
                          max_steps: int = 500):
    """Return a zero-arg constructor compatible with ENV_REGISTRY conventions."""
    def _factory(seed: int = 0, consequence_delay: int = 0, **kw):
        episode_steps = int(kw.pop("max_steps", max_steps))
        return SafetyGymAdapter(
            task_name            = task_name,
            constraint_threshold = constraint_threshold,
            action_dim           = action_dim,
            max_steps            = episode_steps,
            seed                 = seed,
        )
    _factory.__name__ = task_name
    return _factory


# Optional task suite covering several agents and objectives.  Official
# Safety Gymnasium reports costs immediately; these tasks therefore do not by
# themselves validate a delayed-feedback claim.  Task selection is a project
# configuration, not an assertion about benchmark-standard exclusions.
_SAFETY_GYM_TASK_SPECS = {
    "SafetyPointGoal1":   ("SafetyPointGoal1-v0",   25.0, None, 500),
    "SafetyPointGoal2":   ("SafetyPointGoal2-v0",   25.0, None, 500),
    "SafetyCarGoal1":     ("SafetyCarGoal1-v0",      25.0, None, 500),
    "SafetyCarGoal2":     ("SafetyCarGoal2-v0",      25.0, None, 500),
    "SafetyPointButton1": ("SafetyPointButton1-v0",  25.0, None, 500),
    "SafetyPointPush1":   ("SafetyPointPush1-v0",    25.0, None, 500),
    "SafetyAntGoal1":     ("SafetyAntGoal1-v0",      25.0, None, 500),
}

_SAFETY_GYM_AVAILABLE = False
try:
    import safety_gymnasium as _sg_check   # noqa: F401
    _SAFETY_GYM_AVAILABLE = True
except ImportError:
    pass

SAFETY_GYM_ENV_REGISTRY: dict = {}
if _SAFETY_GYM_AVAILABLE:
    for _sg_name, (_sg_task, _sg_thresh, _sg_adim, _sg_steps) in _SAFETY_GYM_TASK_SPECS.items():
        SAFETY_GYM_ENV_REGISTRY[_sg_name] = _make_safety_factory(
            _sg_task, _sg_thresh, _sg_adim, _sg_steps)
    ENV_REGISTRY.update(SAFETY_GYM_ENV_REGISTRY)

SAFETY_GYM_ENVS = tuple(SAFETY_GYM_ENV_REGISTRY.keys())
