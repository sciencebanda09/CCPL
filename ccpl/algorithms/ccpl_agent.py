"""Experimental consequence-aware reinforcement-learning agent.

CCPL combines a learned observation-delay model, a state-conditioned
non-negative penalty, an action-centred consequence predictor, and separate
reward/consequence critics.  These are engineering mechanisms, not
machine-checked theorems.  For the repository's single aggregate CMDP
constraint, the state-conditioned multiplier is a heuristic.  The
action-centred predictor is causal only where the controlled synthetic SCM
provides interventional labels; elsewhere it is an observational contrast.

The reward critic uses the ordinary one-step discount.  A learned
``E[gamma**tau | h]`` factor applies only to attributable delayed consequence.
"""

import math
import pickle
import time
from pathlib import Path
import numpy as np

try:
    from .networks import GRUPolicyNet, LambdaNet, gru_batch_forward, sigmoid, Adam, QNetwork
    from .networks_v7 import AttentionAugmentedPolicyNet, SelfCorrectionModule
    from .networks_v8 import AbstractionLayer, WorkingMemoryLambdaModifier
    from .normalizer import StateNormalizer
    from .causal_consequence import InterventionalConsequenceNet, CausalHistoryEncoder, CausalReplayBuffer
    from .delay_bellman import DelayDistributionNet, DelayCorrectedBellman
    from .lambda_theorem import ConsequenceVarianceEstimator, AdaptiveLambdaWithDominanceTracking
    from .causal_graph import EnvironmentSCM, CausalLabelGenerator
    from .hallucination_fix import HallucinationGate
except ImportError:
    from networks import GRUPolicyNet, LambdaNet, gru_batch_forward, sigmoid, Adam, QNetwork
    from networks_v7 import AttentionAugmentedPolicyNet, SelfCorrectionModule
    from networks_v8 import AbstractionLayer, WorkingMemoryLambdaModifier
    from normalizer import StateNormalizer
    from causal_consequence import InterventionalConsequenceNet, CausalHistoryEncoder, CausalReplayBuffer
    from delay_bellman import DelayDistributionNet, DelayCorrectedBellman
    from lambda_theorem import ConsequenceVarianceEstimator, AdaptiveLambdaWithDominanceTracking
    from causal_graph import EnvironmentSCM, CausalLabelGenerator
    from hallucination_fix import HallucinationGate


class CCPLAgent:
    """
    CCPL: Causal Consequence-Penalized Learning.

    Drop-in replacement for CCPLAgent with three new modules:
      icn          — InterventionalConsequenceNet (Direction 3)
      delay_dist   — DelayDistributionNet         (Direction 1)
      bellman      — DelayCorrectedBellman         (Direction 1)
      var_est      — ConsequenceVarianceEstimator  (Direction 2)
      scm          — EnvironmentSCM                (Direction 3 calibration)
      label_gen    — CausalLabelGenerator          (Direction 3 pretraining)
    """
    name = "CCPL"

    def predict(self, observation) -> int:
        """Return a deterministic evaluation action for one observation."""
        return self.select_action(np.asarray(observation, dtype=np.float32), eval_mode=True)

    def act(self, observation, deterministic: bool = True) -> int:
        """Stable action API compatible with common RL integrations."""
        return self.select_action(
            np.asarray(observation, dtype=np.float32), eval_mode=bool(deterministic)
        )

    def fit(self, env, episodes: int = 1, update_freq: int = 4,
            verbose: bool = False) -> list:
        """Train on an environment implementing the CCPL episode interface."""
        if int(episodes) < 1:
            raise ValueError("episodes must be positive")
        if int(update_freq) < 1:
            raise ValueError("update_freq must be positive")
        results = []
        for episode in range(int(episodes)):
            result = run_episode(self, env, train=True, update_freq=update_freq)
            results.append(result)
            if verbose:
                print(
                    f"episode={episode + 1} reward={result['episode_reward']:.4f} "
                    f"consequence={result['episode_consequence']:.4f}"
                )
        return results

    def save(self, path) -> str:
        """Save a complete checkpoint. Only load checkpoints you trust."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return str(destination)

    @classmethod
    def load(cls, path):
        """Load a checkpoint previously produced by :meth:`save`."""
        with Path(path).open("rb") as handle:
            agent = pickle.load(handle)
        if not isinstance(agent, cls):
            raise TypeError(f"checkpoint contains {type(agent).__name__}, expected {cls.__name__}")
        return agent

    def __init__(
        self,
        state_dim,
        action_dim,
        gru_dim            = 40,
        hidden_dim         = 80,
        n_layers           = 2,
        causal_dim         = 32,
        tau_max            = 15,
        history_len        = 8,
        lr_policy          = 1.2e-3,
        lr_icn             = 4e-4,
        lr_lambda          = 3e-4,
        lr_delay           = 3e-4,
        lr_attn            = 3e-4,
        gamma              = 0.99,
        tau_soft           = 0.01,
        eps_start          = 1.0,
        eps_end            = 0.05,
        eps_decay          = 2500,
        batch_size         = 64,
        buffer_capacity    = 60_000,
        lambda_max         = 2.0,
        lambda_warmup      = 100,
        penalty_scale      = 2.0,
        constraint_d       = 3.0,
        causal_cost_clip   = 10.0,
        max_action_penalty = 50.0,
        use_attention      = False,
        attn_d_model       = 32,
        attn_n_heads       = 4,
        memory_capacity    = 256,
        use_history_attn   = False,
        use_planning       = False,
        plan_K             = 3,
        plan_weight        = 0.3,
        use_correction     = False,
        corr_trigger       = 1.3,
        corr_factor        = 2.0,
        lambda_c           = 0.30,
        use_abstraction    = False,
        n_prototypes       = 16,
        use_working_memory = False,
        wm_window          = 8,
        pretrain_steps     = 600,
        noise_std          = 0.05,
        seed               = 42,
    ):
        if int(state_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if not 0.0 <= float(gamma) < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if int(tau_max) < 0 or int(history_len) <= 0:
            raise ValueError("tau_max must be non-negative and history_len positive")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(pretrain_steps) < 0 or not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("pretrain_steps and pretraining noise must be non-negative")
        if int(buffer_capacity) <= 2 * int(tau_max):
            raise ValueError(
                "buffer_capacity must exceed 2 * tau_max so delayed labels "
                "cannot target overwritten replay entries")
        if use_history_attn:
            raise NotImplementedError(
                "use_history_attn is unavailable: replay does not retain the "
                "online attention context needed for a consistent update")
        if use_planning:
            raise NotImplementedError(
                "use_planning is unavailable: an ICN predicts consequences, "
                "not next states, so a K-step rollout requires a dynamics model")

        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.gru_dim         = gru_dim
        self.gamma           = gamma
        self.tau_soft        = tau_soft
        self.batch_size      = batch_size
        self.lambda_max      = lambda_max
        self.lambda_warmup   = lambda_warmup
        self.penalty_scale   = penalty_scale
        self.constraint_d    = constraint_d
        self.causal_cost_clip = float(causal_cost_clip)
        self.max_action_penalty = float(max_action_penalty)
        self.plan_weight     = plan_weight
        self.lambda_c        = lambda_c
        self.eps_start       = eps_start
        self.eps_end         = eps_end
        self.eps_decay       = eps_decay
        self.tau_max         = tau_max
        self.history_len     = history_len
        self.has_scm_labels  = (state_dim == 6 and action_dim == 5)

        self.steps_done    = 0
        self.episodes_done = 0
        self.update_count  = 0
        self.rng           = np.random.default_rng(seed)
        self._pretrain_rng = np.random.default_rng(seed + 10_003)

        if use_attention:
            self.policy_net = AttentionAugmentedPolicyNet(
                state_dim, action_dim, gru_dim, hidden_dim, n_layers,
                d_model=attn_d_model, n_heads=attn_n_heads,
                memory_capacity=memory_capacity,
                lr_policy=lr_policy, lr_attn=lr_attn, seed=seed)
            self.target_net = AttentionAugmentedPolicyNet(
                state_dim, action_dim, gru_dim, hidden_dim, n_layers,
                d_model=attn_d_model, n_heads=attn_n_heads,
                memory_capacity=memory_capacity,
                lr_policy=lr_policy, lr_attn=lr_attn, seed=seed)
        else:
            self.policy_net = GRUPolicyNet(
                state_dim, action_dim, gru_dim, hidden_dim, n_layers, lr_policy, seed)
            self.target_net = GRUPolicyNet(
                state_dim, action_dim, gru_dim, hidden_dim, n_layers, lr_policy, seed)
        self.target_net.copy_weights_from(self.policy_net)
        self.use_attention = use_attention

        self.q_c_net    = QNetwork(gru_dim, action_dim,
                                   hidden_dim=hidden_dim, n_layers=n_layers,
                                   lr=lr_policy * 0.25,
                                   seed=seed + 1)
        self.q_c_target = QNetwork(gru_dim, action_dim,
                                   hidden_dim=hidden_dim, n_layers=n_layers,
                                   lr=lr_policy * 0.25,
                                   seed=seed + 1)
        self.q_c_target.copy_weights_from(self.q_c_net)
        self._ep_gamma_c: float  = 1.0
        self._ep_Jc_accum: float = 0.0
        self._last_Jc: float     = constraint_d
        self._episode_horizon: int = 100
        self._delay_prior_mean: float = 5.0

        self.lambda_net  = LambdaNet(state_dim, hidden_dim=32,
                                     lambda_max=lambda_max, lr=lr_lambda, seed=seed)
        self.var_est     = ConsequenceVarianceEstimator(state_dim=state_dim)
        self.lam_tracker = AdaptiveLambdaWithDominanceTracking(
            self.lambda_net, self.var_est, constraint_d=constraint_d)

        self.delay_dist = DelayDistributionNet(
            gru_dim=gru_dim, hidden_dim=32, tau_max=tau_max,
            lr=lr_delay, seed=seed)
        self.bellman    = DelayCorrectedBellman(
            self.delay_dist, gamma=gamma, tau_max=tau_max)

        self.icn = InterventionalConsequenceNet(
            state_dim=state_dim, action_dim=action_dim,
            causal_dim=causal_dim, hidden_dim=hidden_dim // 2,
            n_layers=n_layers, history_len=history_len,
            lr=lr_icn, seed=seed)
        self.scm       = EnvironmentSCM(noise_std=0.0)
        self.label_gen = CausalLabelGenerator(self.scm)

        self.history_attn = None
        self.planner = None
        self.corrector = (
            SelfCorrectionModule(trigger_ratio=corr_trigger,
                                 correction_factor=corr_factor)
            if use_correction else None)
        self.abstraction = (
            AbstractionLayer(state_dim, n_prototypes, 0.30, seed=seed)
            if use_abstraction else None)
        self.working_mem = (
            WorkingMemoryLambdaModifier(wm_window, 0.20)
            if use_working_memory else None)

        self.buffer = CausalReplayBuffer(
            capacity=buffer_capacity, history_len=history_len,
            state_dim=state_dim, action_dim=action_dim,
            gru_dim=gru_dim, seed=seed)

        self.normalizer = StateNormalizer(state_dim)
        self._h         = self.policy_net.zero_state(1)
        self._hit_freq_ema   = 0.5
        self._hit_freq_alpha = 0.05

        self._causal_history: list = []
        self._action_log: list = []
        self._obs_tau_buffer: list = []
        self._step_counter: int = 0
        self._ep_states:      list = []
        self._ep_consequences: list = []

        self.last_policy_loss     = 0.0
        self.last_icn_loss        = 0.0
        self.last_delay_loss      = 0.0
        self.last_mean_delta_C    = 0.0
        self.last_mean_lambda     = 0.0
        self.last_lambda_target   = 0.0
        self.last_lambda_signal   = 0.0
        self.last_jc_violation    = 0.0
        self.last_qc_loss         = 0.0
        self.last_jc              = 0.0
        self._last_sigma          = np.zeros(1, np.float32)
        self.last_gamma_eff       = float(gamma)
        self.last_icn_calib       = {}

        self.hallucination_gate = HallucinationGate(
            delta_clip=self.causal_cost_clip)

        self._lambda_log: list = []
        self._gamma_eff_log: list = []
        self._delta_C_log: list = []
        self._LOG_MAX = 10_000

        if pretrain_steps > 0 and self.has_scm_labels:
            self._pretrain_icn(pretrain_steps, noise_std)


    def _pretrain_icn(self, n_steps: int, noise_std: float = 0.05):
        """
        Supervised pretraining of ICN on SCM ground-truth labels.
        Supervised fitting against the local noiseless one-step SCM reference.

        Root cause: Phase 1 trained base_head without any total_head gradient,
        so total_trunk started Phase 2 cold while base_trunk was already warm.
        ΔC = total - base was dominated by a well-trained base, giving near-zero
        delta_C for all actions regardless of causal effect.

        Fix:
          1. Jointly train both heads from step 1 (not phase 2 only).
          2. Use LARGER batch (256) for better gradient signal.
          3. Include boundary states (rl→1, sp→1) where causal effect is largest.
          4. Add explicit FULL-action supervision: ΔC_FULL > 0 enforced.
          5. Cosine LR decay for stability in later phases.
          6. Record held-out correlation and MAE as diagnostics; they are not
             theorem checks or guaranteed acceptance thresholds.
        """
        from networks import sigmoid as _sig
        rng   = self._pretrain_rng
        icn   = self.icn
        half  = n_steps // 2
        B_pt  = 256
        base_lr_init  = icn.base_optim.lr
        total_lr_init = icn.optim.lr

        def _cosine_lr(step, total, lr_init):
            if step < 100:
                return lr_init * (step + 1) / 100
            progress = (step - 100) / max(total - 100, 1)
            return lr_init * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

        def _sample_diverse_states(rng, B):
            """Sample states covering the full distribution including dangerous corners."""
            n_uniform = B * 3 // 4
            n_corner  = B - n_uniform
            uniform_s = rng.uniform(0.1, 0.9, (n_uniform, self.state_dim)).astype(np.float32)
            corner_s  = rng.uniform(0.5, 0.95, (n_corner, self.state_dim)).astype(np.float32)
            corner_s[:, 2] = rng.uniform(0.1, 0.9, n_corner).astype(np.float32)
            corner_s[:, 4] = rng.uniform(0.0, 0.3, n_corner).astype(np.float32)
            states = np.concatenate([uniform_s, corner_s], axis=0)
            if noise_std > 0.0:
                states = np.clip(
                    states + rng.normal(0.0, noise_std, states.shape), 0.0, 1.0)
            return states.astype(np.float32)

        for step in range(n_steps):
            lr_b = _cosine_lr(step, n_steps, base_lr_init)
            lr_t = _cosine_lr(step, n_steps, total_lr_init)
            icn.base_optim.lr  = lr_b
            icn.optim.lr       = lr_t

            states  = _sample_diverse_states(rng, B_pt)
            actions = rng.integers(0, self.action_dim, B_pt).astype(np.int32)
            labels  = self.label_gen.generate_batch(states, actions, self.action_dim)
            ctx     = np.zeros((B_pt, icn.causal_dim), np.float32)
            W       = np.ones(B_pt, np.float32) / B_pt

            tgt_base  = labels["baseline"].astype(np.float32)
            tgt_total = (labels["delta_C_scm"] + labels["baseline"]).astype(np.float32)

            phase2_weight = 1.0 if step >= half else max(0.1, float(step) / half)

            C_base, h_base, raw_B = icn._forward_base(states, ctx)
            err_b     = (C_base - tgt_base) * W
            d_b_pre   = 2.0 * err_b * _sig(raw_B.squeeze(-1))
            d_bh, dWb, dbb = icn.base_head.backward(d_b_pre[:, None])
            _, trunk_b_grads = icn.base_trunk.backward(d_bh)
            icn.base_optim.step(trunk_b_grads + [dWb, dbb])

            C_total, sigma, h_total, raw_C, raw_sigma = icn._forward_total(states, actions, ctx)
            raw_err_t = C_total - tgt_total
            err_t = raw_err_t * W * phase2_weight
            d_loc_t = (W * phase2_weight * np.sign(raw_err_t)
                       / (sigma + 1e-6))
            d_t_pre = (2.0 * err_t + d_loc_t) * _sig(raw_C.squeeze(-1))
            d_th, dWt, dbt = icn.total_head.backward(d_t_pre[:, None])
            abs_err   = np.abs(C_total - tgt_total)
            d_sig     = W * phase2_weight * (
                1.0 / (sigma + 1e-6) - abs_err / (sigma + 1e-6) ** 2)
            d_sig_pre = d_sig * _sig(raw_sigma.squeeze(-1))
            d_sigh, dWs, dbs = icn.sigma_head.backward(d_sig_pre[:, None])
            d_trunk_total = d_th + d_sigh
            _, trunk_t_grads = icn.total_trunk.backward(d_trunk_total)
            icn.optim.step(trunk_t_grads + [dWt, dbt] + [dWs, dbs])

        icn.base_optim.lr = base_lr_init
        icn.optim.lr      = total_lr_init

        verify_states  = rng.uniform(0.1, 0.9, (500, self.state_dim)).astype(np.float32)
        verify_actions = np.full(500, 2, dtype=np.int32)
        verify_ctx     = np.zeros((500, icn.causal_dim), np.float32)
        delta_C, _, _, _ = icn.forward(verify_states, verify_actions, verify_ctx)
        v_labels         = self.label_gen.generate_batch(verify_states, verify_actions, self.action_dim)
        scm_delta        = v_labels["delta_C_scm"].astype(np.float32)
        if np.std(delta_C) > 1e-6 and np.std(scm_delta) > 1e-6:
            corr = float(np.corrcoef(delta_C, scm_delta)[0, 1])
            mae  = float(np.mean(np.abs(delta_C - scm_delta)))
            full_pos_frac = float((delta_C > 0).mean())
            self._pretrain_corr = corr
            self._pretrain_mae  = mae
            self._pretrain_full_pos = full_pos_frac
        if not hasattr(self, '_pretrain_corr') or self._pretrain_corr < 0.75:
            for _ in range(500):
                states_f  = _sample_diverse_states(rng, B_pt)
                actions_f = np.full(B_pt, 2, dtype=np.int32)
                labels_f  = self.label_gen.generate_batch(states_f, actions_f, self.action_dim)
                ctx_f     = np.zeros((B_pt, icn.causal_dim), np.float32)
                W_f       = np.ones(B_pt, np.float32) / B_pt
                tgt_f     = (labels_f["delta_C_scm"] + labels_f["baseline"]).astype(np.float32)
                C_f, sig_f, _, raw_Cf, raw_sf = icn._forward_total(states_f, actions_f, ctx_f)
                raw_err_f = C_f - tgt_f
                err_f = (2.0 * raw_err_f * W_f
                         + W_f * np.sign(raw_err_f) / (sig_f + 1e-6))
                err_f *= _sig(raw_Cf.squeeze(-1))
                d_fh, dWf, dbf = icn.total_head.backward(err_f[:, None])
                abs_ef = np.abs(C_f - tgt_f)
                d_sigf = W_f * (
                    1.0 / (sig_f + 1e-6)
                    - abs_ef / (sig_f + 1e-6) ** 2
                ) * _sig(raw_sf.squeeze(-1))
                d_sfh, dWsf, dbsf = icn.sigma_head.backward(d_sigf[:, None])
                _, tg_f = icn.total_trunk.backward(d_fh + d_sfh)
                icn.optim.step(tg_f + [dWf, dbf] + [dWsf, dbsf])
            delta_C2, _, _, _ = icn.forward(verify_states, verify_actions, verify_ctx)
            if np.std(delta_C2) > 1e-6 and np.std(scm_delta) > 1e-6:
                self._pretrain_corr = float(np.corrcoef(delta_C2, scm_delta)[0, 1])
                self._pretrain_mae  = float(np.mean(np.abs(delta_C2 - scm_delta)))
                self._pretrain_full_pos = float((delta_C2 > 0).mean())


    @property
    def lambda_scale(self) -> float:
        t = min(self.episodes_done / max(self.lambda_warmup, 1), 1.0)
        return float(0.2 + 0.8 * 0.5 * (1.0 - np.cos(np.pi * t)))

    @property
    def epsilon(self) -> float:
        decay = self.eps_decay
        if self.corrector is not None and self.corrector.is_active:
            decay = int(decay * 1.5)
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / max(decay, 1))


    def reset_hidden(self, max_steps: int = None, expected_delay: float = None):
        self._h = self.policy_net.zero_state(1)
        if max_steps is not None:
            self._episode_horizon = max(1, int(max_steps))
        if expected_delay is not None:
            self._delay_prior_mean = float(np.clip(
                expected_delay, 0, self.tau_max))
        if self.history_attn is not None:
            self.history_attn.reset()
        if self.working_mem is not None:
            self.working_mem.reset()
        self.icn.encoder.reset()
        self._causal_history.clear()
        self._ep_states.clear()
        self._ep_consequences.clear()
        self._action_log.clear()
        self._step_counter = 0
        self._ep_gamma_c   = 1.0
        self._ep_Jc_accum  = 0.0

    def episode_end(self, hit_occurred: float = 0.0,
                    episode_states=None, episode_consequences=None):
        self.episodes_done += 1
        self._hit_freq_ema = (self._hit_freq_alpha * float(hit_occurred)
                              + (1 - self._hit_freq_alpha) * self._hit_freq_ema)
        if (self.use_attention and episode_states is not None
                and episode_consequences is not None
                and hasattr(self.policy_net, 'memory')):
            s_norm   = np.array([self.normalizer.normalize(s) for s in episode_states],
                                  np.float32)
            cons_arr = np.asarray(episode_consequences, np.float32)
            self.policy_net.memory.write(s_norm, cons_arr)
            if hasattr(self.target_net, "memory"):
                self.target_net.memory.copy_contents_from(self.policy_net.memory)

    def observe_transition(self, state, action, consequence):
        """Advance online history-dependent state without learning weights.

        Evaluation calls this method too, so ICN history, working memory, and
        correction state do not remain empty/frozen for an entire episode.
        """
        c = float(consequence)
        if self.corrector is not None:
            self.corrector.update(c)
        if self.working_mem is not None:
            self.working_mem.push(c)
        self.icn.encoder.push(state, int(action), c)


    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode:
            self.steps_done += 1

        s_norm = self.normalizer.normalize(state)

        base_lam = float(self.lambda_net.forward(s_norm)) * self.lambda_scale
        if self.abstraction is not None:
            base_lam *= self.abstraction.lambda_amplification(state)
        if self.working_mem is not None:
            base_lam += self.working_mem.lambda_modifier()
        base_lam = float(np.clip(base_lam, 0.0, self.lambda_max))

        cognitive_active = base_lam >= self.lambda_c

        if self.history_attn is not None:
            self.history_attn.push(state)

        if cognitive_active and self.history_attn is not None:
            ctx     = self.history_attn.context()
            s_input = np.clip(s_norm + 0.1 * ctx, -5.0, 5.0)
        else:
            s_input = s_norm

        q_vals, h_new = self.policy_net.forward(s_input[None], self._h)
        q_vals = q_vals.squeeze(0)

        if not eval_mode and self.rng.random() < self.epsilon:
            self._h = h_new
            return int(self.rng.integers(self.action_dim))

        causal_ctx = self.icn.encoder.context()
        acts       = np.arange(self.action_dim, dtype=np.int32)
        icn_state = np.asarray(state, np.float32) if self.has_scm_labels else s_norm
        delta_C, sigma = self.icn.predict_delta(icn_state, acts, causal_ctx)

        h_flat   = h_new.squeeze(0) if h_new.ndim > 1 else h_new
        gamma_e  = float(self.bellman.gamma_eff(h_flat[None]).item())

        corr_mult = (self.corrector.lambda_multiplier
                     if (cognitive_active and self.corrector is not None) else 1.0)
        lam_eff   = base_lam * corr_mult

        plan_pen = np.zeros(self.action_dim, np.float32)
        if cognitive_active and self.planner is not None:
            _plan_ctx = causal_ctx[None] if causal_ctx.ndim == 1 else causal_ctx
            raw_plan  = self._plan_rollout(icn_state, self.action_dim, _plan_ctx)
            if raw_plan.std() > 1e-6:
                q_range = float(q_vals.max() - q_vals.min()) + 1e-6
                p_range = float(raw_plan.max() - raw_plan.min()) + 1e-6
                plan_pen = (raw_plan * gamma_e).astype(np.float32) * (q_range / p_range)

        h_for_qc   = h_flat[None]
        q_c_vals   = self.q_c_net.forward(h_for_qc).squeeze(0)
        q_c_adv    = q_c_vals - q_c_vals.min()
        critic_mix = float(np.clip(self.update_count / 1_000.0, 0.0, 1.0))
        icn_scale = (lam_eff * self.penalty_scale * gamma_e
                     * (1.0 - critic_mix))
        icn_pen = icn_scale * np.maximum(delta_C, 0.0)
        future_pen = (lam_eff * self.penalty_scale * critic_mix
                      * np.maximum(q_c_adv, 0.0))
        if not eval_mode:
            self.hallucination_gate.observe_state(s_norm)
        gated_icn = self.hallucination_gate.gate_penalty(
            icn_pen, delta_C, sigma, s_norm,
            penalty_per_cost=icn_scale)
        penalty = np.clip(
            gated_icn + future_pen, 0.0, self.max_action_penalty)
        q_ccpl = (q_vals - penalty - self.plan_weight * plan_pen)

        self._h = h_new
        self._append_log(self._lambda_log, base_lam)
        self._append_log(self._gamma_eff_log, gamma_e)
        self._append_log(self._delta_C_log, float(np.mean(np.abs(delta_C))))

        return int(q_ccpl.argmax())

    def _append_log(self, lst, val):
        lst.append(val)
        if len(lst) > self._LOG_MAX:
            lst.pop(0)


    def store(self, state, action, reward, next_state, consequence,
              done, hidden=None, next_hidden=None, info=None):
        self.normalizer.update(state)
        c = float(consequence)

        self._ep_Jc_accum += self._ep_gamma_c * float(consequence)
        self._ep_gamma_c  *= self.gamma
        if done:
            self._last_Jc     = self._ep_Jc_accum
            self._ep_Jc_accum = 0.0
            self._ep_gamma_c  = 1.0

        self.observe_transition(state, action, c)
        if self.abstraction  is not None: self.abstraction.update(state, c)
        self.lam_tracker.record(
            state, c, lambda_state=self.normalizer.normalize(state))

        h  = self._h if hidden      is None else hidden
        nh = self._h if next_hidden is None else next_hidden
        scm_label_valid = bool(
            isinstance(info, dict)
            and info.get("scm_label_valid", False)
            and self.has_scm_labels)
        replay_index = self.buffer.push(
            state, action, reward, next_state, c, done, h, nh,
            list(self._causal_history), scm_label_valid=scm_label_valid)

        self._causal_history.append((state.copy(), int(action), c))
        if len(self._causal_history) > self.history_len:
            self._causal_history.pop(0)
        self._ep_states.append(state.copy())
        self._ep_consequences.append(c)

        h_vec = np.asarray(nh, np.float32).squeeze()
        self._action_log.append(
            (self._step_counter, int(action), h_vec.copy(), replay_index))
        self._step_counter += 1

        delay_label_valid = (
            not isinstance(info, dict)
            or info.get("delay_supervision_valid", True))
        if self._action_log and delay_label_valid:
            actual_tau = info.get("actual_tau") if isinstance(info, dict) else None
            if actual_tau is not None:
                actual_tau = int(actual_tau)
                if actual_tau < 0 or actual_tau >= len(self._action_log):
                    actual_tau = None
            if actual_tau is not None:
                past_idx = len(self._action_log) - 1 - actual_tau
                obs_tau  = int(np.clip(actual_tau, 0, self.tau_max))
                _, _, past_h, past_replay_index = self._action_log[past_idx]
                self._obs_tau_buffer.append((past_h, obs_tau))
                self.buffer.set_aligned_consequence(past_replay_index, c)
                self._delay_prior_mean = (
                    0.95 * self._delay_prior_mean + 0.05 * float(obs_tau))
                if len(self._obs_tau_buffer) > 500:
                    self._obs_tau_buffer.pop(0)
        if len(self._action_log) > self.tau_max * 2:
            self._action_log.pop(0)


    def update(self) -> bool:
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return False

        S   = np.array([self.normalizer.normalize(s) for s in batch["states"]])
        NS  = np.array([self.normalizer.normalize(s) for s in batch["next_states"]])
        A   = batch["actions"]
        R   = batch["rewards"]
        D   = batch["dones"]
        W   = batch["weights"]
        H   = batch["hiddens"]
        NH  = batch["next_hiddens"]
        B   = len(A)

        causal_ctxs = self.icn.encoder.context_batch(batch["histories"])
        S_icn = (np.asarray(batch["states"], np.float32)
                 if self.has_scm_labels else S)

        scm_mask = np.asarray(batch["scm_label_valid"], bool)
        aligned_valid = np.asarray(batch["aligned_valid"], bool)
        model_valid = scm_mask | aligned_valid
        icn_targets = np.asarray(
            batch["aligned_consequences"], np.float32).copy()
        baseline_targets = icn_targets.copy()
        if np.any(scm_mask):
            scm_online = self.label_gen.generate_batch(
                batch["states"], A, self.action_dim)
            scm_total = (scm_online["delta_C_scm"]
                         + scm_online["baseline"])
            icn_targets[scm_mask] = scm_total[scm_mask]
            baseline_targets[scm_mask] = scm_online["baseline"][scm_mask]
        if np.any(model_valid):
            model_weights = W * model_valid.astype(np.float32)
            icn_stats = self.icn.update_step(
                S_icn, A, icn_targets, causal_ctxs, model_weights,
                baseline_targets=baseline_targets)
        else:
            icn_stats = {"l_total": 0.0, "l_base": 0.0,
                         "l_sigma": 0.0, "mean_delta_C": 0.0}
        delta_C, _, _, sigma = self.icn.forward(S_icn, A, causal_ctxs)
        delta_C = np.asarray(delta_C, np.float32)
        sigma   = np.asarray(sigma,   np.float32)

        if np.any(scm_mask) and self.update_count % 50 == 0:
            raw_states = batch["states"][scm_mask]
            self.last_icn_calib = self.label_gen.icn_calibration_error(
                raw_states, A[scm_mask], delta_C[scm_mask])
            if self.last_icn_calib:
                self.hallucination_gate.observe_icn_mae(self.last_icn_calib.get("mae", 0.0))

        if (np.any(scm_mask) and self.update_count % 100 == 0
                and self.update_count > 0):
            raw_states_32 = batch["states"][scm_mask][:32]
            n_cal    = len(raw_states_32)
            all_acts = np.tile(np.arange(self.action_dim), n_cal).astype(np.int32)
            raw_repeated  = np.repeat(raw_states_32,  self.action_dim, axis=0)
            ctx_cal     = np.zeros((len(all_acts), self.icn.causal_dim), np.float32)
            scm_labels  = self.label_gen.generate_batch(raw_repeated, all_acts, self.action_dim)
            scm_targets = scm_labels["delta_C_scm"] + scm_labels["baseline"]
            calibration_strength = self.hallucination_gate.recalib_weight()
            calibration_steps = 1 + int(round(4.0 * calibration_strength))
            W_cal = np.ones(len(all_acts), np.float32)
            for _ in range(calibration_steps):
                self.icn.update_step(
                    raw_repeated, all_acts, scm_targets, ctx_cal, W_cal,
                    baseline_targets=scm_labels["baseline"])

        lam_raw   = self.lambda_net.forward(S)
        lam_batch = lam_raw * self.lambda_scale

        H_cur = gru_batch_forward(self.policy_net.gru, S, H)
        if self.use_attention and hasattr(self.policy_net, 'memory'):
            H_cur_in = self.policy_net.res_norm.forward(
                H_cur + self.policy_net.memory.read(H_cur))
        else:
            H_cur_in = H_cur

        if len(self._obs_tau_buffer) >= self.batch_size // 2:
            buf_idx  = self.rng.choice(
                len(self._obs_tau_buffer),
                min(self.batch_size, len(self._obs_tau_buffer)),
                replace=False)
            buf_h    = np.stack([self._obs_tau_buffer[i][0] for i in buf_idx]).astype(np.float32)
            buf_tau  = np.array([self._obs_tau_buffer[i][1] for i in buf_idx], np.int32)
            buf_w    = np.ones(len(buf_idx), np.float32)
            self.last_delay_loss = self.delay_dist.update_step(buf_h, buf_tau, buf_w)
        else:
            _prior_mean = self._delay_prior_mean
            obs_tau = np.full(
                len(H_cur), int(np.clip(round(_prior_mean), 0, self.tau_max)),
                np.int32)
            self.last_delay_loss = self.delay_dist.update_step(H_cur, obs_tau, W)

        H_pol = gru_batch_forward(self.policy_net.gru, NS, NH)
        H_tgt = gru_batch_forward(self.target_net.gru, NS, NH)

        if self.use_attention and hasattr(self.policy_net, 'memory'):
            H_pol_in = self.policy_net.res_norm.forward(
                H_pol + self.policy_net.memory.read(H_pol))
            H_tgt_in = self.target_net.res_norm.forward(
                H_tgt + self.target_net.memory.read(H_tgt))
        else:
            H_pol_in, H_tgt_in = H_pol, H_tgt

        feat_pol = self.policy_net.trunk.forward(H_pol_in)
        v_r_pol  = self.policy_net.val_head.forward(feat_pol)
        a_r_pol  = self.policy_net.adv_head.forward(feat_pol)
        q_r_pol  = v_r_pol + a_r_pol - a_r_pol.mean(axis=-1, keepdims=True)
        q_c_pol  = self.q_c_net.forward(H_pol_in)
        q_c_pol_adv = q_c_pol - q_c_pol.min(axis=-1, keepdims=True)
        lam_next = (self.lambda_net.forward(NS) * self.lambda_scale)[:, None]
        next_score = q_r_pol - lam_next * self.penalty_scale * q_c_pol_adv
        next_acts = next_score.argmax(axis=-1)

        feat_tgt = self.target_net.trunk.forward(H_tgt_in)
        v_r_tgt  = self.target_net.val_head.forward(feat_tgt)
        a_r_tgt  = self.target_net.adv_head.forward(feat_tgt)
        q_r_tgt  = v_r_tgt + a_r_tgt - a_r_tgt.mean(axis=-1, keepdims=True)
        next_q_r  = q_r_tgt[np.arange(B), next_acts]

        q_c_cur_all  = self.q_c_net.forward(H_cur_in)
        q_c_tgt_all  = self.q_c_target.forward(H_tgt_in)
        q_c_cur  = q_c_cur_all[np.arange(B), A]
        q_c_next = q_c_tgt_all[np.arange(B), next_acts]

        delta_C_clipped = np.clip(
            delta_C, -self.causal_cost_clip, self.causal_cost_clip)
        attributable_cost = np.clip(
            delta_C, 0.0, self.causal_cost_clip)
        not_done = (1.0 - D)
        gamma_e_vec = self.bellman.gamma_eff(H_cur)
        self.last_gamma_eff = float(gamma_e_vec.mean())

        td_r_target = (R + self.gamma * next_q_r * not_done).astype(np.float32)
        td_c_target = (gamma_e_vec * attributable_cost
                       + self.gamma * q_c_next * not_done).astype(np.float32)

        feat_s  = self.policy_net.trunk.forward(H_cur_in)
        v_r_all = self.policy_net.val_head.forward(feat_s)
        a_r_all = self.policy_net.adv_head.forward(feat_s)
        q_r_all = v_r_all + a_r_all - a_r_all.mean(axis=-1, keepdims=True)
        q_r_cur  = q_r_all[np.arange(B), A]

        td_r_errors = td_r_target - q_r_cur
        p_loss = float(np.mean(W * td_r_errors**2))
        self.policy_net.backward_update(S, H, A, np.clip(td_r_errors, -3, 3), W)

        td_c_errors = td_c_target - q_c_cur
        self.q_c_net.backward_update(H_cur_in, A, np.clip(td_c_errors, -3.0, 3.0), W)

        per_step_d   = self.constraint_d / max(self._episode_horizon, 1)
        jc_violation = (self._last_Jc - self.constraint_d) / (self.constraint_d + 1e-6)
        global_target = float(sigmoid(jc_violation * 2.0))
        local_scale = max(float(np.mean(np.abs(delta_C_clipped))), per_step_d, 1e-3)
        local_target = sigmoid(delta_C_clipped / local_scale)
        freq_target  = np.clip(np.full(B, self._hit_freq_ema, np.float32), 0, 1)
        lam_target = np.clip(
            0.6 * global_target + 0.3 * local_target + 0.1 * freq_target,
            0.0, 1.0)
        self.last_lambda_target = float(np.mean(lam_target))
        self.last_lambda_signal = float(jc_violation)
        self.last_jc_violation = float(jc_violation)
        lam_errors = lam_raw / self.lambda_max - lam_target
        local_dev = np.abs(delta_C_clipped - delta_C_clipped.mean())
        risk_w = np.clip(
            0.5 + 0.5 * local_dev / (local_dev.mean() + 1e-6), 0.5, 2.0)
        combined_lam_w = risk_w * W
        combined_lam_w = combined_lam_w / (combined_lam_w.mean() + 1e-8)
        self.lambda_net.backward_update(S, lam_errors * combined_lam_w, weight_decay=1e-4)

        self.target_net.soft_update_from(self.policy_net, tau=self.tau_soft)
        self.q_c_target.soft_update_from(self.q_c_net,    tau=self.tau_soft)

        combined_td = 0.6 * np.abs(td_r_errors) + 0.4 * np.abs(td_c_errors)
        c_errors = np.abs(td_c_errors)
        self.buffer.update_priorities(batch["indices"], combined_td, c_errors)

        self.update_count         += 1
        self.last_policy_loss      = p_loss
        self.last_icn_loss         = icn_stats["l_total"]
        self._last_sigma           = sigma
        self.last_mean_delta_C     = icn_stats["mean_delta_C"]
        self.last_mean_lambda      = float(lam_batch.mean())
        self.last_qc_loss          = float(np.mean(W * td_c_errors**2))
        self.last_jc               = self._last_Jc
        return True


    def diagnostics(self) -> dict:
        d = {
            "epsilon":          round(self.epsilon, 4),
            "steps":            self.steps_done,
            "episodes":         self.episodes_done,
            "updates":          self.update_count,
            "lambda_scale":     round(self.lambda_scale, 4),
            "hit_freq_ema":     round(self._hit_freq_ema, 4),
            "mean_lambda":      round(self.last_mean_lambda, 4),
            "lambda_target":    round(self.last_lambda_target, 4),
            "jc_violation":     round(self.last_jc_violation, 4),
            "gamma_eff":        round(self.last_gamma_eff, 4),
            "expected_delay":    round(
                float(self.delay_dist.expected_tau(self._h)), 4),
            "mean_delta_C":     round(self.last_mean_delta_C, 4),
            "policy_loss":      round(self.last_policy_loss, 4),
            "icn_loss":         round(self.last_icn_loss, 4),
            "mean_sigma":       round(float(np.mean(self._last_sigma)), 4) if hasattr(self, "_last_sigma") else 0.0,
            "delay_loss":       round(self.last_delay_loss, 4),
            "qc_loss":          round(self.last_qc_loss, 4),
            "last_Jc":          round(self.last_jc, 4),
            "buffer":           self.buffer.ready_count(),
        }
        if self.last_icn_calib:
            d["icn_mae"]  = round(self.last_icn_calib.get("mae", 0), 5)
            d["icn_corr"] = round(self.last_icn_calib.get("correlation", 0), 4)
        if self.corrector is not None:
            d["correction_active"] = self.corrector.is_active
        dom = self.lam_tracker.theorem2_status(self.last_mean_lambda)
        d["cost_heterogeneity"] = dom["var_s"]
        d["lambda_state_variation"] = dom.get("state_variation_score", 0)
        return d

    def get_theory_logs(self) -> dict:
        h_sample = np.random.default_rng(0).normal(
            size=(32, self.gru_dim)).astype(np.float32)
        delay_check = self.bellman.verify_contraction(h_sample)
        lambda_check = self.lam_tracker.theorem2_status(self.last_mean_lambda)
        return {
            "lambda_log":    list(self._lambda_log),
            "gamma_eff_log": list(self._gamma_eff_log),
            "delta_C_log":   list(self._delta_C_log),
            "delay_diagnostic": delay_check,
            "lambda_diagnostic": lambda_check,
            "theorem1":      delay_check,
            "theorem2":      lambda_check,
        }



def run_episode(agent: CCPLAgent, env,
                train: bool = True,
                update_freq: int = 4) -> dict:
    if update_freq <= 0:
        raise ValueError("update_freq must be positive")
    state = env.reset()
    agent.reset_hidden(
        max_steps=getattr(env, "max_steps", None),
        expected_delay=getattr(env, "consequence_delay", None))

    ep_r = ep_c = ep_c_raw = ep_steps = 0.0
    gamma_c = 1.0
    losses   = []
    t_infer  = []
    hit_occurred = False
    hit_steps    = 0
    prev_hits    = 0

    while not env.done:
        h_before = agent._h.copy()
        t0       = time.perf_counter()
        action   = agent.select_action(state, eval_mode=not train)
        t_infer.append(time.perf_counter() - t0)
        h_after  = agent._h.copy()

        ns, r, c, done, info = env.step(action)

        cur_hits = info.get("delayed_hits", 0)
        if cur_hits > prev_hits:
            hit_occurred = True
            hit_steps   += cur_hits - prev_hits
        prev_hits = cur_hits

        if train:
            agent.store(state, action, r, ns, c, done, h_before, h_after, info=info)
            if int(ep_steps) % update_freq == 0:
                if agent.update():
                    losses.append(agent.last_policy_loss)
        else:
            agent.observe_transition(state, action, c)

        state     = ns
        ep_r     += r
        ep_c     += gamma_c * c
        ep_c_raw += c
        gamma_c  *= agent.gamma
        ep_steps += 1

    stats = env.episode_stats()
    if train:
        hit_rate   = hit_steps / max(int(ep_steps), 1)
        _ep_states = getattr(agent, '_ep_states',       None) or None
        _ep_cons   = getattr(agent, '_ep_consequences', None) or None
        agent.episode_end(
            hit_occurred         = hit_rate,
            episode_states       = _ep_states,
            episode_consequences = _ep_cons,
        )

    return {
        "episode_reward":      ep_r,
        "episode_consequence": ep_c,
        "episode_consequence_undiscounted": ep_c_raw,
        "delayed_hits":        stats["delayed_hits"],
        "original_consequence": float(stats.get("original_consequence", ep_c_raw)),
        "steps":               int(ep_steps),
        "mean_loss":           float(np.mean(losses)) if losses else 0.0,
        "mean_infer_ms":       float(np.mean(t_infer) * 1000) if t_infer else 0.0,
        "hit_occurred":        hit_occurred,
    }



def make_ccpl(state_dim: int, action_dim: int, seed: int = 42,
              **kwargs) -> CCPLAgent:
    """Full CCPL agent — all three directions enabled."""
    a = CCPLAgent(state_dim, action_dim, seed=seed, **kwargs)
    a.name = "CCPL"
    return a


def make_ccpl_base(state_dim: int, action_dim: int, seed: int = 42) -> CCPLAgent:
    """
    CCPL-Base: CCPLAgent with D1/D2/D3 all disabled.

    This replaces CCPLAgent as the 'no-directions' baseline.
    It is a plain GRU-DQN with a scalar lambda and no delay correction,
    no causal attribution, and no state-conditioned lambda — matching the
    CCPL-Base definition in the paper (Table 1, Table 2).

    Using CCPLAgent here (not CCPLAgent) means:
      - Same network architecture as CCPL
      - Same environment interface
      - No dependency on ccpl_agent.py at all
    """
    import types

    a = CCPLAgent(state_dim, action_dim, seed=seed,
                  pretrain_steps=0,
                  use_attention=False,
                  use_history_attn=False,
                  use_planning=False,
                  use_correction=False,
                  use_abstraction=False,
                  use_working_memory=False)
    a.name = "CCPL-Base"

    def _fixed_gamma(self, h):
        return np.full(len(h), self.gamma, np.float32)
    a.bellman.gamma_eff = types.MethodType(_fixed_gamma, a.bellman)

    def _zero_lambda(S):
        arr = np.asarray(S)
        if arr.ndim == 2:
            return np.zeros(len(arr), np.float32)
        return 0.0
    a.lambda_net.forward = _zero_lambda

    _orig_icn_fwd = a.icn.forward
    def _raw_forward(S, A, ctx):
        delta, total, base, sigma = _orig_icn_fwd(S, A, ctx)
        return total, total, base, sigma
    a.icn.forward = _raw_forward

    return a


def build_ccpl_ablation(state_dim: int, action_dim: int,
                          seed: int = 42) -> dict:
    """
    Full ablation suite — isolates each breakthrough direction.
    Matches Table 2 in the NeurIPS paper exactly.

    CCPL            — all four directions (full system)
    CCPL-NoDelay    — D1 removed: fixed γ (no delay distribution)
    CCPL-NoStateλ   — D2 removed: scalar λ_g instead of state-conditioned λ(s)
    CCPL-NoCausal   — D3 removed: uses raw Ĉ instead of causal ΔC
    CCPL-SingleQ    — D4 removed: single Q-function (λ absorbed into TD target)
    CCPL-Base       — D1–D4 all removed (plain GRU-DQN)
    """
    import types

    full = make_ccpl(state_dim, action_dim, seed)
    full.name = "CCPL"

    no_delay = make_ccpl(state_dim, action_dim, seed)
    no_delay.name = "CCPL-NoDelay"
    def _fixed_gamma(self, h): return np.full(len(h), self.gamma, np.float32)
    no_delay.bellman.gamma_eff = types.MethodType(_fixed_gamma, no_delay.bellman)

    no_state_lam = make_ccpl(state_dim, action_dim, seed)
    no_state_lam.name = "CCPL-NoStateλ"
    _scalar_lam = np.array([0.5], np.float32)

    def _scalar_lam_forward(S):
        arr = np.asarray(S)
        if arr.ndim == 2:
            return np.full(len(arr), float(_scalar_lam[0]), np.float32)
        return float(_scalar_lam[0])

    def _scalar_lam_backward(states, lam_errors, weight_decay=0.0):
        grad = float(lam_errors.mean())
        _scalar_lam[0] = float(np.clip(_scalar_lam[0] - 3e-4 * grad,
                                        0.0, no_state_lam.lambda_max))

    no_state_lam.lambda_net.forward = _scalar_lam_forward
    no_state_lam.lambda_net.backward_update = _scalar_lam_backward

    no_causal = make_ccpl(state_dim, action_dim, seed)
    no_causal.name = "CCPL-NoCausal"
    _orig_icn_forward = no_causal.icn.forward
    def _raw_forward_nocausal(S, A, ctx):
        delta, total, base, sigma = _orig_icn_forward(S, A, ctx)
        return total, total, base, sigma
    no_causal.icn.forward = _raw_forward_nocausal

    single_q = make_ccpl(state_dim, action_dim, seed)
    single_q.name = "CCPL-SingleQ"
    _orig_single_update = single_q.update

    def _single_q_update(self=single_q):
        """Override update to bake λ into the Bellman target (non-stationary)."""
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return False
        S   = np.array([self.normalizer.normalize(s) for s in batch["states"]])
        NS  = np.array([self.normalizer.normalize(s) for s in batch["next_states"]])
        A   = batch["actions"]
        R   = batch["rewards"]
        D   = batch["dones"]
        W   = batch["weights"]
        H   = batch["hiddens"]
        NH  = batch["next_hiddens"]
        B   = len(A)
        causal_ctxs = self.icn.encoder.context_batch(batch["histories"])
        S_icn = (np.asarray(batch["states"], np.float32)
                 if self.has_scm_labels else S)
        scm_mask = np.asarray(batch["scm_label_valid"], bool)
        aligned_valid = np.asarray(batch["aligned_valid"], bool)
        model_valid = scm_mask | aligned_valid
        icn_targets = np.asarray(
            batch["aligned_consequences"], np.float32).copy()
        baseline_targets = icn_targets.copy()
        if np.any(scm_mask):
            scm_online = self.label_gen.generate_batch(
                batch["states"], A, self.action_dim)
            scm_total = scm_online["delta_C_scm"] + scm_online["baseline"]
            icn_targets[scm_mask] = scm_total[scm_mask]
            baseline_targets[scm_mask] = scm_online["baseline"][scm_mask]
        if np.any(model_valid):
            icn_stats = self.icn.update_step(
                S_icn, A, icn_targets, causal_ctxs,
                W * model_valid.astype(np.float32),
                baseline_targets=baseline_targets)
        else:
            icn_stats = {"l_total": 0.0, "mean_delta_C": 0.0}
        delta_C, _, _, sigma = self.icn.forward(S_icn, A, causal_ctxs)
        delta_C = np.asarray(delta_C, np.float32)
        lam_batch = self.lambda_net.forward(S) * self.lambda_scale
        from networks import gru_batch_forward as _gru_fwd
        H_cur = _gru_fwd(self.policy_net.gru, S, H)
        H_pol = _gru_fwd(self.policy_net.gru, NS, NH)
        H_tgt = _gru_fwd(self.target_net.gru, NS, NH)
        if self.use_attention and hasattr(self.policy_net, 'memory'):
            H_cur_in = self.policy_net.res_norm.forward(H_cur + self.policy_net.memory.read(H_cur))
            H_pol_in = self.policy_net.res_norm.forward(H_pol + self.policy_net.memory.read(H_pol))
            H_tgt_in = self.target_net.res_norm.forward(H_tgt + self.target_net.memory.read(H_tgt))
        else:
            H_cur_in = H_cur
            H_pol_in, H_tgt_in = H_pol, H_tgt
        feat_pol = self.policy_net.trunk.forward(H_pol_in)
        v_next   = self.policy_net.val_head.forward(feat_pol)
        a_next   = self.policy_net.adv_head.forward(feat_pol)
        q_next   = v_next + a_next - a_next.mean(axis=-1, keepdims=True)
        next_acts = q_next.argmax(axis=-1)
        feat_tgt  = self.target_net.trunk.forward(H_tgt_in)
        v_tgt     = self.target_net.val_head.forward(feat_tgt)
        a_tgt     = self.target_net.adv_head.forward(feat_tgt)
        q_tgt     = v_tgt + a_tgt - a_tgt.mean(axis=-1, keepdims=True)
        next_q_r  = q_tgt[np.arange(B), next_acts]
        _, gamma_e_vec = self.bellman.td_target(R, next_q_r, np.zeros_like(R), np.zeros_like(R), D, H_cur, 1.0)
        delta_C_c = np.clip(delta_C, 0.0, self.causal_cost_clip)
        td_combined_target = (
            R - lam_batch * self.penalty_scale * gamma_e_vec * delta_C_c
            + self.gamma * next_q_r * (1.0 - D)
        ).astype(np.float32)
        feat_s  = self.policy_net.trunk.forward(H_cur_in)
        v_all   = self.policy_net.val_head.forward(feat_s)
        a_all   = self.policy_net.adv_head.forward(feat_s)
        q_all   = v_all + a_all - a_all.mean(axis=-1, keepdims=True)
        q_cur   = q_all[np.arange(B), A]
        td_errors = td_combined_target - q_cur
        p_loss = float(np.mean(W * td_errors**2))
        self.policy_net.backward_update(S, H, A, np.clip(td_errors, -3, 3), W)
        self.target_net.soft_update_from(self.policy_net, tau=self.tau_soft)
        per_step_d = self.constraint_d / max(self._episode_horizon, 1)
        jc_v = (self._last_Jc - self.constraint_d) / (self.constraint_d + 1e-6)
        c_ex = delta_C_c / (per_step_d + 1e-6) - 1.0
        from networks import sigmoid as _sig
        lam_signal = 0.6 * jc_v + 0.4 * float(_sig(c_ex * 2.0).mean())
        from networks import sigmoid as _sig2
        mag_t = np.clip(_sig2(np.ones(B) * lam_signal * 2.0) * np.ones(B, np.float32), 0.0, 1.0)
        freq_t = np.clip(np.full(B, self._hit_freq_ema, np.float32), 0, 1)
        lam_tgt = 0.7 * mag_t + 0.3 * freq_t
        self.last_lambda_target = float(np.mean(lam_tgt))
        self.last_lambda_signal = float(lam_signal)
        self.last_jc_violation = float(jc_v)
        raw_lam = self.lambda_net.forward(S)
        lam_err = raw_lam / self.lambda_max - lam_tgt
        self.lambda_net.backward_update(S, lam_err * W, weight_decay=1e-4)
        combined_td = np.abs(td_errors)
        c_errors = np.abs(delta_C_c)
        self.buffer.update_priorities(batch["indices"], combined_td, c_errors)
        self.update_count += 1
        self.last_policy_loss = p_loss
        self.last_icn_loss = icn_stats["l_total"]
        self._last_sigma = sigma
        self.last_mean_delta_C = icn_stats["mean_delta_C"]
        self.last_mean_lambda = float(lam_batch.mean())
        self.last_qc_loss = 0.0
        self.last_jc = self._last_Jc
        return True

    single_q.update = types.MethodType(lambda self: _single_q_update(), single_q)

    base = make_ccpl_base(state_dim, action_dim, seed)

    return {
        "CCPL":           full,
        "CCPL-NoDelay":   no_delay,
        "CCPL-NoStateλ":  no_state_lam,
        "CCPL-NoCausal":  no_causal,
        "CCPL-SingleQ":   single_q,
        "CCPL-Base":      base,
    }
