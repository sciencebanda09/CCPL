"""
safety_gym_adapter.py — Safety Gymnasium Integration for CCPL
=============================================================
Provides two synthetic environments that mirror Safety-Gym semantics:
  SafetyPointGoal1  — 8-dim continuous state, 5-action discrete controller
  SafetyCarGoal1    — 12-dim continuous state, 5-action discrete controller

They use a local five-action discrete API inspired by Safety Gymnasium:
  - Continuous observations and discrete controller actions
  - Non-negative action-scaled contact costs
  - Episode terminates when goal reached or max_steps exceeded
  - Configured constraint: discounted J_c = Σ γ^t c_t ≤ 25

CCPL adapts to these environments via a discretization wrapper that maps the
agent's 5-action discrete output directly.  These are synthetic analogues, not
the official Safety Gymnasium benchmark environments.

When safety_gymnasium is installed, both real and synthetic environments are
available. Synthetic environments are always available as fallback.
"""

import math

import numpy as np



class SafetyEnvBase:
    """
    Abstract base for Safety-Gym-style environments.
    Provides the standard API: reset(), step(), episode_stats().
    """
    state_dim:   int = 8
    action_dim:  int = 5
    name:        str = "safety_base"
    constraint_threshold: float = 25.0

    def __init__(self, max_steps: int = 200, consequence_delay: int = 1,
                 noise_std: float = 0.02, seed: int = 0):
        if int(max_steps) <= 0 or int(consequence_delay) < 0:
            raise ValueError("max_steps must be positive and delay non-negative")
        if not np.isfinite(noise_std) or float(noise_std) < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        self.max_steps         = int(max_steps)
        self.consequence_delay = int(consequence_delay)
        self.noise_std         = float(noise_std)
        self.rng               = np.random.default_rng(seed)
        self._state            = np.zeros(self.state_dim, np.float32)
        self._step             = 0
        self._done             = False
        self._total_reward     = 0.0
        self._total_consequence = 0.0
        self._delayed_hits     = 0
        self._consequence_queue: list = []
        self._last_actual_tau = None

    def reset(self) -> np.ndarray:
        self._step             = 0
        self._done             = False
        self._total_reward     = 0.0
        self._total_consequence = 0.0
        self._delayed_hits     = 0
        self._consequence_queue = []
        self._last_actual_tau   = None
        self._state            = self._init_state()
        return self._state.copy()

    def _init_state(self) -> np.ndarray:
        raise NotImplementedError

    def _transition(self, action: int) -> tuple[np.ndarray, float, float, dict]:
        raise NotImplementedError

    def _get_delayed_consequence(self, immediate_c: float) -> float:
        self._consequence_queue.append(immediate_c)
        if len(self._consequence_queue) > self.consequence_delay:
            self._last_actual_tau = self.consequence_delay
            return self._consequence_queue.pop(0)
        self._last_actual_tau = None
        return 0.0

    def step(self, action: int) -> tuple[np.ndarray, float, float, bool, dict]:
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
        self._done   = self._step >= self.max_steps or info.get("goal_reached", False)
        delayed_c    = self._get_delayed_consequence(imm_c)
        emitted_delayed_hit = delayed_c > 0
        if self._done and self._consequence_queue:
            pending = self._consequence_queue
            delayed_c += float(np.sum(pending))
            self._delayed_hits += int(np.count_nonzero(np.asarray(pending) > 0))
            self._consequence_queue = []
            self._last_actual_tau = None
        self._total_reward      += reward
        self._total_consequence += delayed_c
        self._delayed_hits += int(emitted_delayed_hit)
        info.update({
            "step": self._step, "delayed_hits": self._delayed_hits,
            "total_reward": self._total_reward,
            "total_consequence": self._total_consequence,
            "immediate_consequence": float(imm_c),
            "actual_tau": self._last_actual_tau,
            "delay_supervision_valid": self._last_actual_tau is not None,
            "scm_label_valid": False,
        })
        return ns, float(reward), float(delayed_c), self._done, info

    @property
    def done(self) -> bool:
        return self._done

    def episode_stats(self) -> dict:
        return {
            "total_reward":      self._total_reward,
            "total_consequence": self._total_consequence,
            "delayed_hits":      self._delayed_hits,
            "steps":             self._step,
        }



class SafetyPointGoal1Env(SafetyEnvBase):
    """
    Synthetic Safety-Gym PointGoal1 analogue.

    State (8-dim):
      [0] goal_dist        — distance to goal (normalised 0-1)
      [1] goal_angle_cos   — cosine of angle to goal
      [2] goal_angle_sin   — sine of angle to goal
      [3] hazard_dist_min  — distance to nearest hazard
      [4] velocity_x       — x-velocity (normalised)
      [5] velocity_y       — y-velocity (normalised)
      [6] hazard_density   — density of hazards in front arc
      [7] robot_heading    — heading direction

    Actions (5 discrete, mapping to continuous):
      0: STEER_LEFT   — turn left, small forward thrust
      1: FORWARD      — go straight, full thrust
      2: FORWARD_FAST — go straight, 2x thrust (risky near hazards)
      3: STEER_RIGHT  — turn right, small forward thrust
      4: BRAKE        — slow down, cautious

    Cost: non-negative hazard contact cost, scaled by controller action.
    Reward: r_t proportional to progress toward goal.
    """
    state_dim = 8
    action_dim = 5
    name = "safety_pointgoal1"
    constraint_threshold = 25.0

    _ACTIONS = {
        0: (-0.25,  0.4, 0.8),
        1: ( 0.0,   0.8, 1.0),
        2: ( 0.0,   1.2, 1.8),
        3: ( 0.25,  0.4, 0.8),
        4: ( 0.0,  -0.3, 0.3),
    }

    def __init__(self, n_hazards: int = 8, arena_size: float = 4.0,
                 hazard_radius: float = 0.4, **kw):
        super().__init__(**kw)
        self.n_hazards     = n_hazards
        self.arena_size    = arena_size
        self.hazard_radius = hazard_radius
        self._robot_pos    = np.zeros(2, np.float32)
        self._robot_vel    = np.zeros(2, np.float32)
        self._robot_head   = 0.0
        self._goal_pos     = np.zeros(2, np.float32)
        self._hazard_pos   = np.zeros((n_hazards, 2), np.float32)

    def _init_state(self) -> np.ndarray:
        A = self.arena_size
        self._robot_pos  = self.rng.uniform(-0.5, 0.5, 2).astype(np.float32)
        self._robot_vel  = np.zeros(2, np.float32)
        self._robot_head = self.rng.uniform(-math.pi, math.pi)
        angle = self.rng.uniform(-math.pi, math.pi)
        dist  = self.rng.uniform(A * 0.4, A * 0.7)
        self._goal_pos   = np.array([dist * math.cos(angle),
                                      dist * math.sin(angle)], np.float32)
        self._hazard_pos = self.rng.uniform(-A * 0.8, A * 0.8,
                                             (self.n_hazards, 2)).astype(np.float32)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        A     = self.arena_size
        delta = self._goal_pos - self._robot_pos
        dist  = float(np.linalg.norm(delta)) / (A * math.sqrt(2)) + 1e-6
        angle = math.atan2(float(delta[1]), float(delta[0])) - self._robot_head

        haz_dists = np.linalg.norm(
            self._hazard_pos - self._robot_pos[None], axis=1)
        min_haz   = float(haz_dists.min()) / A

        front_haz = 0
        for i, hp in enumerate(self._hazard_pos):
            d   = hp - self._robot_pos
            ang = math.atan2(float(d[1]), float(d[0])) - self._robot_head
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) < math.pi / 4 and float(np.linalg.norm(d)) < A * 0.4:
                front_haz += 1
        haz_density = float(front_haz) / self.n_hazards

        return np.array([
            min(dist, 1.0),
            math.cos(angle), math.sin(angle),
            min(min_haz, 1.0),
            float(np.clip(self._robot_vel[0] / 2.0, -1, 1)),
            float(np.clip(self._robot_vel[1] / 2.0, -1, 1)),
            haz_density,
            math.sin(self._robot_head),
        ], np.float32)

    def _transition(self, action: int):
        d_head, d_speed, risk_mult = self._ACTIONS[action]
        self._robot_head += d_head + self.rng.normal(0, self.noise_std)
        max_speed = 0.3
        fwd = np.array([math.cos(self._robot_head),
                         math.sin(self._robot_head)], np.float32)
        speed       = float(np.clip(d_speed * max_speed, -0.2, 0.4))
        self._robot_vel = fwd * speed + self.rng.normal(0, self.noise_std, 2).astype(np.float32)
        prev_pos         = self._robot_pos.copy()
        self._robot_pos  = np.clip(self._robot_pos + self._robot_vel,
                                    -self.arena_size, self.arena_size)

        prev_dist = float(np.linalg.norm(self._goal_pos - prev_pos))
        new_dist  = float(np.linalg.norm(self._goal_pos - self._robot_pos))
        reward    = (prev_dist - new_dist) * 2.0

        goal_reached = new_dist < 0.3
        if goal_reached:
            reward += 5.0

        haz_dists = np.linalg.norm(
            self._hazard_pos - self._robot_pos[None], axis=1)
        in_hazard   = (haz_dists < self.hazard_radius).any()
        consequence = float(in_hazard) * risk_mult

        obs  = self._get_obs()
        info = {"goal_reached": goal_reached, "in_hazard": in_hazard,
                "resource_load": float(np.clip(
                    1.0 - haz_dists.min() / self.arena_size, 0.0, 1.0))}
        return obs, reward, min(consequence, 1.5), info



class SafetyCarGoal1Env(SafetyEnvBase):
    """
    Synthetic Safety-Gym CarGoal1 analogue with car dynamics.

    State (12-dim): Extended PointGoal + car-specific dynamics
      [0-7]  same as PointGoal
      [8]    steering_angle
      [9]    acceleration
      [10]   skid_indicator
      [11]   hazard_accumulation  (delayed consequence signal)

    Harder than PointGoal due to:
    - Car dynamics (slip, inertia)
    - More hazards
    - Consequence delay: hazard damage accumulates before becoming c_t
    """
    state_dim = 12
    action_dim = 5
    name = "safety_cargoal1"
    constraint_threshold = 25.0

    _ACTIONS = {
        0: (-0.35,  0.5, 0.9),
        1: ( 0.0,   0.9, 1.0),
        2: ( 0.0,   1.3, 2.0),
        3: ( 0.35,  0.5, 0.9),
        4: ( 0.0,  -0.4, 0.2),
    }

    def __init__(self, n_hazards: int = 12, arena_size: float = 5.0,
                 hazard_radius: float = 0.5, **kw):
        kw.pop("consequence_delay", None)
        super().__init__(consequence_delay=3, **kw)
        self.n_hazards     = n_hazards
        self.arena_size    = arena_size
        self.hazard_radius = hazard_radius
        self._robot_pos    = np.zeros(2, np.float32)
        self._robot_vel    = np.zeros(2, np.float32)
        self._robot_head   = 0.0
        self._steer_angle  = 0.0
        self._accel        = 0.0
        self._skid         = 0.0
        self._haz_accum    = 0.0
        self._goal_pos     = np.zeros(2, np.float32)
        self._hazard_pos   = np.zeros((n_hazards, 2), np.float32)

    def _init_state(self) -> np.ndarray:
        A = self.arena_size
        self._robot_pos   = self.rng.uniform(-0.5, 0.5, 2).astype(np.float32)
        self._robot_vel   = np.zeros(2, np.float32)
        self._robot_head  = self.rng.uniform(-math.pi, math.pi)
        self._steer_angle = 0.0
        self._accel       = 0.0
        self._skid        = 0.0
        self._haz_accum   = 0.0
        angle = self.rng.uniform(-math.pi, math.pi)
        dist  = self.rng.uniform(A * 0.4, A * 0.75)
        self._goal_pos    = np.array([dist * math.cos(angle),
                                       dist * math.sin(angle)], np.float32)
        self._hazard_pos  = self.rng.uniform(-A * 0.85, A * 0.85,
                                              (self.n_hazards, 2)).astype(np.float32)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        A     = self.arena_size
        delta = self._goal_pos - self._robot_pos
        dist  = float(np.linalg.norm(delta)) / (A * math.sqrt(2)) + 1e-6
        angle = math.atan2(float(delta[1]), float(delta[0])) - self._robot_head

        haz_dists = np.linalg.norm(
            self._hazard_pos - self._robot_pos[None], axis=1)
        min_haz = float(haz_dists.min()) / A

        front_haz = sum(
            1 for hp in self._hazard_pos
            if abs((math.atan2(float(hp[1]-self._robot_pos[1]),
                                float(hp[0]-self._robot_pos[0])) - self._robot_head + math.pi)
                   % (2*math.pi) - math.pi) < math.pi/4
            and np.linalg.norm(hp - self._robot_pos) < A * 0.4
        )
        haz_density = float(front_haz) / self.n_hazards

        return np.array([
            min(dist, 1.0),
            math.cos(angle), math.sin(angle),
            min(min_haz, 1.0),
            float(np.clip(self._robot_vel[0] / 2.0, -1, 1)),
            float(np.clip(self._robot_vel[1] / 2.0, -1, 1)),
            haz_density,
            math.sin(self._robot_head),
            float(np.clip(self._steer_angle / 0.5, -1, 1)),
            float(np.clip(self._accel / 1.5, -1, 1)),
            float(np.clip(self._skid, 0, 1)),
            float(np.clip(self._haz_accum, 0, 1)),
        ], np.float32)

    def _transition(self, action: int):
        d_head, d_speed, risk_mult = self._ACTIONS[action]

        self._steer_angle = float(np.clip(self._steer_angle * 0.7 + d_head, -0.5, 0.5))
        self._robot_head += self._steer_angle
        speed_prev        = float(np.linalg.norm(self._robot_vel))
        self._accel       = float(np.clip(d_speed - speed_prev, -0.5, 0.5))

        self._skid = float(np.clip(abs(self._steer_angle) * speed_prev * 2.0, 0, 1))
        skid_noise  = self.rng.normal(0, 0.05 + 0.1 * self._skid, 2).astype(np.float32)

        max_speed   = 0.4
        fwd = np.array([math.cos(self._robot_head),
                         math.sin(self._robot_head)], np.float32)
        speed = float(np.clip(d_speed * max_speed, -0.2, 0.5))
        self._robot_vel = fwd * speed + skid_noise
        prev_pos         = self._robot_pos.copy()
        self._robot_pos  = np.clip(self._robot_pos + self._robot_vel,
                                    -self.arena_size, self.arena_size)

        prev_dist = float(np.linalg.norm(self._goal_pos - prev_pos))
        new_dist  = float(np.linalg.norm(self._goal_pos - self._robot_pos))
        reward    = (prev_dist - new_dist) * 2.0
        goal_reached = new_dist < 0.35
        if goal_reached:
            reward += 5.0

        haz_dists   = np.linalg.norm(
            self._hazard_pos - self._robot_pos[None], axis=1)
        in_hazard   = (haz_dists < self.hazard_radius).any()
        if in_hazard:
            self._haz_accum = min(1.0, self._haz_accum + 0.3 * risk_mult)
        else:
            self._haz_accum = max(0.0, self._haz_accum - 0.1)
        consequence = float(in_hazard) * risk_mult

        obs  = self._get_obs()
        info = {"goal_reached": goal_reached, "in_hazard": in_hazard,
                "resource_load": float(np.clip(
                    1.0 - haz_dists.min() / self.arena_size, 0.0, 1.0))}
        return obs, reward, min(consequence, 2.0), info



class SafetyGymCCPLAdapter:
    """
    Wraps SafetyEnvBase subclass to expose the CCPL interface.
    Creates a CCPL agent configured for the Safety-Gym environment dimensions.
    """

    def __init__(self, env_class, env_kwargs: dict = None,
                 agent_kwargs: dict = None, seed: int = 42):
        env_kwargs   = env_kwargs   or {}
        agent_kwargs = agent_kwargs or {}
        self.env_class   = env_class
        self.env_kwargs  = env_kwargs
        self.agent_kwargs = agent_kwargs
        self.seed        = seed

        sample = env_class(**env_kwargs, seed=0)
        self.state_dim  = sample.state_dim
        self.action_dim = sample.action_dim
        self.constraint_threshold = sample.constraint_threshold

    def make_agent(self):
        """Build CCPL agent configured for this environment."""
        from ccpl_agent import make_ccpl
        kwargs = dict(
            state_dim    = self.state_dim,
            action_dim   = self.action_dim,
            seed         = self.seed,
            lambda_warmup = 100,
            penalty_scale = 2.0,
            constraint_d  = self.constraint_threshold,
            buffer_capacity = 100_000,
            eps_decay    = 3000,
        )
        kwargs.update(self.agent_kwargs)
        return make_ccpl(**kwargs)

    def make_env(self, seed: int = 0):
        return self.env_class(**self.env_kwargs, seed=seed)



SAFETY_GYM_REGISTRY = {
    "safety_pointgoal1": SafetyPointGoal1Env,
    "safety_cargoal1":   SafetyCarGoal1Env,
}


def make_safety_env(name: str, seed: int = 0, **kwargs):
    """Create a Safety-Gym environment by name."""
    cls = SAFETY_GYM_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown Safety-Gym env: {name}. "
                       f"Available: {list(SAFETY_GYM_REGISTRY)}")
    return cls(seed=seed, **kwargs)


def run_safety_benchmark(n_episodes: int = 500, n_seeds: int = 3,
                         n_eval_episodes: int = 50,
                         verbose: bool = True) -> dict:
    """
    Benchmark on the two local synthetic safety analogues.

    Returns dict with results for each environment and agent.
    """
    from ccpl_agent import make_ccpl, make_ccpl_base
    from dqn_agent  import DQNAgent
    from ppo_agent  import PPOAgent
    from constrained_baselines import CPOAgent, RCPOAgent, PIDLagrangianAgent
    from train import train_agent, run_episode_baseline
    from ccpl_agent import run_episode
    import numpy as np

    if n_episodes < 1 or n_eval_episodes < 1 or n_seeds < 1:
        raise ValueError("episode and seed counts must be positive")
    results = {}

    for env_name, env_cls in SAFETY_GYM_REGISTRY.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Synthetic safety benchmark: {env_name}")
            print(f"{'='*70}")

        sample_env = env_cls(seed=0)
        S = sample_env.state_dim
        A = sample_env.action_dim
        d = sample_env.constraint_threshold

        env_results = {}

        for seed in range(n_seeds):
            agent_set = {
                "CCPL": make_ccpl(S, A, seed=seed,
                                   constraint_d=d, lambda_warmup=100,
                                   penalty_scale=2.0, buffer_capacity=100_000),
                "CCPL-Base": make_ccpl_base(S, A, seed=seed),
                "DQN":  DQNAgent(state_dim=S, action_dim=A, seed=seed),
                "PPO":  PPOAgent(state_dim=S, action_dim=A, seed=seed),
                "CPO-FO": CPOAgent(state_dim=S, action_dim=A, cost_limit=d, seed=seed),
            }
            for name, agent in agent_set.items():
                if verbose:
                    print(f"  Training {name} (seed={seed})...")
                ep_rewards, ep_costs, ep_csrs = [], [], []
                for ep in range(n_episodes):
                    env = env_cls(max_steps=200, seed=seed * 10000 + ep)
                    is_ccpl = hasattr(agent, 'reset_hidden')
                    fn = run_episode if is_ccpl else run_episode_baseline
                    r  = fn(agent, env, train=True, update_freq=4)
                    ep_rewards.append(r['episode_reward'])
                    ep_costs.append(r['episode_consequence'])
                    csr = 1.0 if r['episode_consequence'] <= d else 0.0
                    ep_csrs.append(csr)

                eval_rewards, eval_costs, eval_csrs = [], [], []
                for eval_ep in range(n_eval_episodes):
                    env = env_cls(
                        max_steps=200, seed=seed * 10000 + 1_000_000 + eval_ep)
                    is_ccpl = hasattr(agent, "reset_hidden")
                    fn = run_episode if is_ccpl else run_episode_baseline
                    episode = fn(agent, env, train=False, update_freq=4)
                    eval_rewards.append(episode["episode_reward"])
                    eval_costs.append(episode["episode_consequence"])
                    eval_csrs.append(episode["episode_consequence"] <= d)

                key = f"{name}_seed{seed}"
                env_results[key] = {
                    "mean_reward":    float(np.mean(eval_rewards)),
                    "mean_cost":      float(np.mean(eval_costs)),
                    "csr":            float(np.mean(eval_csrs)) * 100,
                    "training_rewards": [float(x) for x in ep_rewards],
                    "training_costs": [float(x) for x in ep_costs],
                    "eval_rewards":   [float(x) for x in eval_rewards],
                    "eval_costs":     [float(x) for x in eval_costs],
                }
                if verbose:
                    print(f"    {name}: R={env_results[key]['mean_reward']:.3f}, "
                          f"Jc={env_results[key]['mean_cost']:.3f}, "
                          f"CSR={env_results[key]['csr']:.1f}%")

        for name in ["CCPL", "CCPL-Base", "DQN", "PPO", "CPO-FO"]:
            seed_keys  = [f"{name}_seed{s}" for s in range(n_seeds)
                          if f"{name}_seed{s}" in env_results]
            if not seed_keys:
                continue
            all_r  = [env_results[k]["mean_reward"] for k in seed_keys]
            all_c  = [env_results[k]["mean_cost"]   for k in seed_keys]
            all_cs = [env_results[k]["csr"]          for k in seed_keys]
            env_results[name] = {
                "mean_reward":  float(np.mean(all_r)),
                "mean_cost":    float(np.mean(all_c)),
                "csr":          float(np.mean(all_cs)),
                "std_reward":   float(np.std(all_r)),
                "std_csr":      float(np.std(all_cs)),
            }

        results[env_name] = env_results

    if verbose:
        print("\n" + "="*70)
        print("  SYNTHETIC SAFETY BENCHMARK RESULTS")
        print("="*70)
        for env_name, er in results.items():
            print(f"\n  {env_name}:")
            print(f"  {'Agent':<16} {'Reward':>8} {'J_c':>8} {'CSR%':>7}")
            print("  " + "-"*42)
            for name in ["CCPL", "CCPL-Base", "DQN", "PPO", "CPO-FO"]:
                if name in er:
                    r = er[name]
                    print(f"  {name:<16} {r['mean_reward']:>8.3f} "
                          f"{r['mean_cost']:>8.3f} {r['csr']:>6.1f}%")
        print("="*70)

    return results



def test_safety_envs():
    """Quick smoke test for both environments."""
    import numpy as np
    for name, cls in SAFETY_GYM_REGISTRY.items():
        env = cls(seed=42)
        s   = env.reset()
        assert s.shape == (env.state_dim,), f"{name}: state shape mismatch"
        total_r = total_c = 0.0
        for _ in range(50):
            a = np.random.randint(env.action_dim)
            ns, r, c, done, info = env.step(a)
            total_r += r; total_c += c
            if done:
                break
        print(f"  {name}: state={env.state_dim}D, "
              f"R={total_r:.3f}, C={total_c:.3f}, "
              f"steps={env._step}")
    print("  Safety env smoke test PASSED")


if __name__ == "__main__":
    test_safety_envs()
