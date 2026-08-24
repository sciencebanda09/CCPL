"""
causal_graph.py — Structural Causal Model of the CCPL Environment
==================================================================
DIRECTION 3: A one-step structural reference model for the synthetic simulator.

The deterministic part of the environment's causal structure is specified by
``environments.py``.  The model below omits transition noise and evaluates an
immediate action intervention at a fixed state.  It is not an identified causal
model for external environments or a long-horizon counterfactual simulator.
We use this to:
  1. Generate ground-truth causal attribution labels for training the ICN
  2. Compute counterfactual consequences: "what would happen if I took a different action?"
  3. Identify causal pathways: which state variables carry the consequence?

THE CAUSAL GRAPH of BaseEnv:

  a_t ──────────────────────────────────────────────────────► r_t
         │                                                    │
         │    rl_t ──────────────────────────────────────► c_t
         │     ▲           ▲                                  │
         ├────►│    fr_t ──┤◄─ a_t                           │
         │     │    sp_t ──┘                                  │
         │     │    hpl_t ──────────────────────────────────► c_t
         │     │     ▲                                         
         └────►rl_{t+1}                                        
               fr_{t+1} ◄── a_t                               
               sp_{t+1} ◄── a_t                               
               hpl_{t+1} ◄── a_t, fr_t   ← HIDDEN PENALTY PATH

  CRITICAL: hpl (hidden_penalty) is updated as:
    dhpl = 0.15*fr   when  action == FULL (2)
    
  This means FULL action accumulates hidden penalties PROPORTIONAL to future_risk.
  An agent without causal understanding treats high hpl as a property of the state,
  not a consequence of its own past FULL actions. CCPL fixes this.

STRUCTURAL EQUATIONS (from _transition):
  
  rl_{t+1} = clip(rl_t + drl(a) + ε₁,  0, 1)
  fr_{t+1} = clip(fr_t + dfr(a) + ε₂,  0, 1)
  ap_{t+1} = Uniform(0.1, 0.9)                    ← exogenous
  sp_{t+1} = clip(sp_t + dsp(a) + ε₄,  0, 1)
  unc_{t+1} = clip(unc_t + ε₅,          0, 1)    ← random walk
  hpl_{t+1} = clip(hpl_t + dhpl(a, fr) + ε₆, 0, 1)

  consequence = 0.5·F + 0.3·U + 0.2·D
    F = drl·fr + hpl·1{a=FULL}·0.5    ← CAUSAL PATH: a → F → c
    U = max(0, rl-0.6)/0.4 + drl·max(0, rl-0.5)
    D = sp·1{a=FULL}·0.6 + unc·|dsp|·0.3

COUNTERFACTUAL ENGINE:
  
  Given observed (s, a, c), compute:
    c_cf(a') = consequence if action a' had been taken instead of a
    ΔC_cf(s, a) = c - E_{a'~Uniform}[c_cf(a')]   ← counterfactual attribution
  
  This gives noiseless, model-generated one-step reference labels for the ICN.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Structural Causal Model
# ─────────────────────────────────────────────────────────────────────────────

class EnvironmentSCM:
    """
    Structural Causal Model of the CCPL base environment.

    Provides noiseless one-step counterfactual consequences under the local
    synthetic equations. This is used to:
      1. Generate training labels for InterventionalConsequenceNet
      2. Verify that ICN approximations converge to SCM ground truth
      3. Diagnose ICN agreement with its synthetic reference labels
    """

    # Deterministic immediate-cost terms from BaseEnv._transition().
    _ACTION_EFFECTS = {
        0: dict(drl=-0.05, dfr= 0.08,  dsp= 0.05, dhpl_coeff=0.0),   # DEFER
        1: dict(drl= 0.10, dfr=-0.04,  dsp=-0.03, dhpl_coeff=0.0),   # PARTIAL
        2: dict(drl= 0.25, dfr=-0.10,  dsp=-0.06, dhpl_coeff=0.15),  # FULL
        3: dict(drl= 0.12, dfr=-0.20,  dsp=-0.08, dhpl_coeff=0.0),   # INVEST
        4: dict(drl=-0.10, dfr= 0.02,  dsp=-0.15, dhpl_coeff=0.0),   # REBALANCE
    }

    def __init__(self, noise_std: float = 0.0):
        """
        Only the deterministic structural reference is implemented.  The
        argument remains for compatibility and must be zero.
        """
        if float(noise_std) != 0.0:
            raise NotImplementedError(
                "Stochastic counterfactual sampling requires shared exogenous "
                "noise and is not implemented")
        self.noise_std = 0.0

    def _reward(self, action: int, ap: float, rl: float) -> float:
        if   action == 0: return 0.10 * ap
        elif action == 1: return 0.30 * ap + 0.10 * (1 - rl)
        elif action == 2: return 0.70 * ap + 0.20 * (1 - rl)
        elif action == 3: return -0.15
        else:             return 0.05

    def _consequence(self, state: np.ndarray, action: int,
                     noise: np.ndarray | None = None) -> float:
        """
        Compute consequence for a given (state, action) pair using the SCM.
        If noise is None, uses noiseless (deterministic) counterfactual.
        """
        rl, fr, ap, sp, unc, hpl = state.astype(np.float64)
        fx = self._ACTION_EFFECTS[action]
        drl, dsp = fx["drl"], fx["dsp"]

        if noise is not None:
            rl_new = np.clip(rl + drl + noise[0], 0, 1)
        else:
            rl_new = np.clip(rl + drl, 0, 1)

        F = np.clip(drl * fr + hpl * float(action == 2) * 0.5, 0, 1)
        U = np.clip(max(0, rl_new - 0.6) / 0.4 + drl * max(0, rl - 0.5), 0, 1)
        D = np.clip(sp * float(action == 2) * 0.6 + unc * abs(dsp) * 0.3, 0, 1)
        return float(0.5 * F + 0.3 * U + 0.2 * D)

    def counterfactual(self, state: np.ndarray, actual_action: int,
                       counterfactual_action: int) -> float:
        """
        c_cf(a') = consequence if a' had been taken instead of actual_action.
        Uses the noiseless one-step structural equations.
        """
        return self._consequence(state, counterfactual_action, noise=None)

    def causal_attribution(self, state: np.ndarray, action: int) -> float:
        """
        ΔC_scm(s, a) = c(s, a) - (1/|A|) Σ_{a'} c(s, a')
        
        The marginal consequence of action a relative to the uniform baseline.
        This is a model-generated reference contrast, not external ground truth.
        """
        c_actual  = self._consequence(state, action)
        c_mean_cf = np.mean([self._consequence(state, a)
                             for a in range(len(self._ACTION_EFFECTS))])
        return float(c_actual - c_mean_cf)

    def causal_attribution_all(self, state: np.ndarray) -> np.ndarray:
        """
        Returns ΔC_scm(s, a) for all actions simultaneously.
        Shape: (action_dim,)
        """
        n_actions = len(self._ACTION_EFFECTS)
        c_vals    = np.array([self._consequence(state, a) for a in range(n_actions)],
                              np.float32)
        return (c_vals - c_vals.mean()).astype(np.float32)

    def most_causal_action(self, state: np.ndarray) -> int:
        """Returns the action with the highest causal consequence contribution."""
        attrs = self.causal_attribution_all(state)
        return int(np.argmax(attrs))

    def safe_action(self, state: np.ndarray) -> int:
        """Returns the action with the LOWEST causal consequence contribution."""
        attrs = self.causal_attribution_all(state)
        return int(np.argmin(attrs))

    def causal_chain(self, state: np.ndarray, action: int) -> dict:
        """
        Decompose the consequence into its three causal pathways:
          F-path: drl·fr + hpl·1{FULL}·0.5    (future-risk amplification)
          U-path: resource overload penalty
          D-path: system pressure × uncertainty
        """
        rl, fr, ap, sp, unc, hpl = state.astype(np.float64)
        fx  = self._ACTION_EFFECTS[action]
        drl = fx["drl"]
        dsp = fx["dsp"]

        F = float(np.clip(drl * fr + hpl * float(action == 2) * 0.5, 0, 1))
        rl_new = np.clip(rl + drl, 0, 1)
        U = float(np.clip(max(0, rl_new - 0.6) / 0.4 + drl * max(0, rl - 0.5), 0, 1))
        D = float(np.clip(sp * float(action == 2) * 0.6 + unc * abs(dsp) * 0.3, 0, 1))

        c_total = 0.5 * F + 0.3 * U + 0.2 * D
        return {
            "F":       round(F, 4),
            "U":       round(U, 4),
            "D":       round(D, 4),
            "c_total": round(c_total, 4),
            "F_share": round(0.5 * F / (c_total + 1e-8), 3),
            "U_share": round(0.3 * U / (c_total + 1e-8), 3),
            "D_share": round(0.2 * D / (c_total + 1e-8), 3),
            "dominant_pathway": "F" if F > U and F > D else ("U" if U > D else "D"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ICN Ground Truth Label Generator
# ─────────────────────────────────────────────────────────────────────────────

class CausalLabelGenerator:
    """
    Generates ground-truth causal attribution labels using the SCM.
    Used to:
      1. Pretrain the ICN before RL training (supervised phase)
      2. Evaluate ICN accuracy during training (ICN vs SCM agreement)
      3. Diagnose which actions are causally dangerous vs just correlated
    """

    def __init__(self, scm: EnvironmentSCM = None):
        self.scm = scm if scm is not None else EnvironmentSCM(noise_std=0.0)

    def generate_batch(self, states: np.ndarray, actions: np.ndarray,
                       n_actions: int = 5) -> dict:
        """
        Generate SCM causal attribution labels for a batch of (state, action) pairs.
        For non-6-dim states (e.g. Safety-Gym), falls back to zero labels.

        Returns:
          delta_C_scm    : (B,) synthetic reference contrast for each (s, a)
          delta_C_all    : (B, n_actions) attribution for all actions (for ICN calibration)
          baseline       : (B,) E_{a'}[c(s, a')] — consequence baseline
          dominant_path  : (B,) str — 'F', 'U', or 'D' dominant pathway
        """
        B = len(states)
        if states.ndim != 2 or len(actions) != B:
            raise ValueError("states must be 2-D and actions must match its batch size")
        delta_C_scm  = np.zeros(B, np.float32)
        delta_C_all  = np.zeros((B, n_actions), np.float32)
        baseline     = np.zeros(B, np.float32)

        # Guard: SCM only applies to the 6-dim base environment
        if states.shape[1] != 6:
            return {"delta_C_scm": delta_C_scm,
                    "delta_C_all": delta_C_all,
                    "baseline":    baseline}
        if n_actions != len(self.scm._ACTION_EFFECTS):
            raise ValueError("The synthetic SCM is defined for exactly five actions")

        for i in range(B):
            s          = states[i]
            a          = int(actions[i])
            attrs_all  = self.scm.causal_attribution_all(s)
            c_vals     = np.array([self.scm._consequence(s, aa)
                                   for aa in range(n_actions)], np.float32)
            baseline[i]    = c_vals.mean()
            delta_C_scm[i] = float(attrs_all[a])
            delta_C_all[i] = attrs_all

        return {
            "delta_C_scm":  delta_C_scm,
            "delta_C_all":  delta_C_all,
            "baseline":     baseline,
        }

    def icn_calibration_error(self, states: np.ndarray, actions: np.ndarray,
                               icn_delta_C: np.ndarray) -> dict:
        """
        Measure how well the ICN approximates the SCM ground truth.
        Returns MAE, correlation, and per-action breakdown.
        """
        labels = self.generate_batch(states, actions)
        scm_dc = labels["delta_C_scm"]
        mae    = float(np.abs(icn_delta_C - scm_dc).mean())
        if (len(icn_delta_C) > 1 and np.std(icn_delta_C) > 1e-12
                and np.std(scm_dc) > 1e-12):
            corr = float(np.corrcoef(icn_delta_C, scm_dc)[0, 1])
        else:
            corr = 0.0
        return {
            "mae":         round(mae, 5),
            "correlation": round(corr, 4),
            "well_calibrated": mae < 0.05 and corr > 0.7,
        }
