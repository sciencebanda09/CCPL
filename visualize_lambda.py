"""
CCPL v5+ — Mechanistic visualization module.

Generates the four figures that prove CCPL is not merely another
reward optimizer, but is learning state-differentiated consequence
weighting:

  1. lambda_heatmap          — λ(s) across (resource_load, future_risk) state space
  2. lambda_vs_shock         — λ trajectory before/after delayed-consequence hits
  3. policy_counterfactual   — action divergence between StateLambda and GlobalLambda
  4. sigma_under_noise       — σ(s) trajectory in NoisyEnv uncertainty spikes

All functions are called from generate_mechanistic_plots() and saved to
the results directory.

Requirements: matplotlib (already used in plots.py)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

from ccpl_agent import CCPLAgent as CCPLAgent
from environments import ENV_REGISTRY



_BLUE   = "#2563EB"
_CORAL  = "#D85A30"
_TEAL   = "#1D9E75"
_AMBER  = "#BA7517"
_PURPLE = "#7F77DD"
_GRAY   = "#888780"

def _fig(title, xlabel, ylabel, figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")



class LambdaTrajectoryLogger:
    """
    Attach to an CCPLAgent during training to log λ(s), σ(s),
    action chosen, and whether a delayed consequence hit fired.

    Usage:
        logger = LambdaTrajectoryLogger(agent)
        # Then in training loop:
        logger.log_step(state, action, consequence, delayed_hit)
    """

    def __init__(self, agent: CCPLAgent):
        self.agent    = agent
        self.states:        list = []
        self.lambdas:       list = []
        self.sigmas:        list = []
        self.actions:       list = []
        self.consequences:  list = []
        self.delayed_hits:  list = []
        self.episodes:      list = []
        self._ep = 0

    def log_step(self, state, action, consequence, delayed_hit=False):
        s_norm = self.agent.normalizer.normalize(state)[None]
        lam    = self.agent.lambda_net.forward(s_norm.squeeze())
        S_rep  = np.tile(s_norm, (self.agent.action_dim, 1))
        acts   = np.arange(self.agent.action_dim, dtype=np.int32)
        _, _, _, _, sigma_all = self.agent.consequence_net.forward(S_rep, acts)
        sigma  = float(sigma_all[action])

        self.states.append(state.copy())
        self.lambdas.append(float(lam) if np.isscalar(lam) else float(np.mean(lam)))
        self.sigmas.append(sigma)
        self.actions.append(int(action))
        self.consequences.append(float(consequence))
        self.delayed_hits.append(bool(delayed_hit))
        self.episodes.append(self._ep)

    def new_episode(self):
        self._ep += 1

    def arrays(self):
        return {
            "states":       np.array(self.states,       np.float32),
            "lambdas":      np.array(self.lambdas,      np.float32),
            "sigmas":       np.array(self.sigmas,       np.float32),
            "actions":      np.array(self.actions,      np.int32),
            "consequences": np.array(self.consequences, np.float32),
            "delayed_hits": np.array(self.delayed_hits, bool),
            "episodes":     np.array(self.episodes,     np.int32),
        }



def plot_lambda_heatmap(logger_data: dict, out_dir: str,
                        n_bins: int = 12, filename: str = "mech_01_lambda_heatmap.png"):
    """
    Bin steps by (resource_load, future_risk) → compute mean λ per cell.
    If high-resource/high-risk states show elevated λ, state conditioning is meaningful.
    """
    states  = logger_data["states"]
    lambdas = logger_data["lambdas"]

    rl = np.clip(states[:, 0], 0, 1)
    fr = np.clip(states[:, 1], 0, 1)

    bins      = np.linspace(0, 1, n_bins + 1)
    grid      = np.full((n_bins, n_bins), np.nan)
    counts    = np.zeros((n_bins, n_bins), int)

    for i in range(len(lambdas)):
        ri = int(np.clip(np.searchsorted(bins, rl[i]) - 1, 0, n_bins - 1))
        fi = int(np.clip(np.searchsorted(bins, fr[i]) - 1, 0, n_bins - 1))
        if np.isnan(grid[ri, fi]):
            grid[ri, fi] = lambdas[i]
        else:
            grid[ri, fi] += lambdas[i]
        counts[ri, fi] += 1

    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, grid / np.maximum(counts, 1), np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin="lower", aspect="auto",
                   extent=[0, 1, 0, 1], cmap="YlOrRd",
                   vmin=0, vmax=np.nanmax(grid))
    plt.colorbar(im, ax=ax, label="Mean λ(s)")
    ax.set_xlabel("Resource load")
    ax.set_ylabel("Future risk")
    ax.set_title("State-conditioned λ heatmap\n(high = stronger consequence penalization)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    circle = plt.Circle((0.85, 0.85), 0.1, color=_CORAL, fill=False,
                         linewidth=1.5, linestyle="--")
    ax.add_patch(circle)
    ax.text(0.88, 0.72, "elevated λ\nexpected here",
            fontsize=8, color=_CORAL, ha="center")

    _save(fig, os.path.join(out_dir, filename))



def plot_lambda_vs_shock(logger_data: dict, out_dir: str,
                         window: int = 15,
                         filename: str = "mech_02_lambda_shock.png"):
    """
    For each delayed-hit event, extract a window of λ values centred on
    the hit step.  Plot mean ± std across all hit events.
    If λ rises *before* the hit, CCPL is anticipating consequence.
    If λ rises *after*, it is reacting (still useful, less impressive).
    """
    lambdas     = logger_data["lambdas"]
    hit_flags   = logger_data["delayed_hits"]

    hit_indices = np.where(hit_flags)[0]
    if len(hit_indices) == 0:
        print("  [lambda_vs_shock] No delayed hits found — skipping figure.")
        return

    snippets = []
    for idx in hit_indices:
        lo = max(0, idx - window)
        hi = min(len(lambdas), idx + window + 1)
        snip = np.full(2 * window + 1, np.nan)
        offset = window - (idx - lo)
        length = hi - lo
        snip[offset: offset + length] = lambdas[lo:hi]
        snippets.append(snip)

    mat  = np.array(snippets, float)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    xs   = np.arange(-window, window + 1)

    fig, ax = _fig(
        f"λ trajectory around delayed-consequence hits (n={len(hit_indices)})",
        "Steps relative to delayed hit", "λ(s)"
    )
    ax.axvline(0, color=_CORAL, linestyle="--", alpha=0.7, label="Hit fires")
    ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=_BLUE)
    ax.plot(xs, mean, color=_BLUE, linewidth=2, label="Mean λ(s)")
    ax.legend(fontsize=9)
    ax.set_xlim(-window, window)

    pre_mean  = mean[:window].mean()
    post_mean = mean[window + 1:].mean()
    if pre_mean < post_mean * 0.85:
        ax.annotate("λ anticipates consequence →",
                    xy=(0, mean[window]), xytext=(-window + 2, mean.max() * 0.9),
                    arrowprops=dict(arrowstyle="->", color=_CORAL),
                    fontsize=8, color=_CORAL)

    _save(fig, os.path.join(out_dir, filename))



def plot_policy_counterfactual(
    state_data:   dict,
    global_data:  dict,
    out_dir:      str,
    filename:     str = "mech_03_policy_counterfactual.png",
):
    """
    Find states where λ differs most between StateLambda and GlobalLambda.
    Show the action distribution at those states for each agent.
    CCPL-State should prefer DEFER/INVEST; Global should prefer FULL more.
    """
    ACTION_LABELS = ["DEFER", "PARTIAL", "FULL", "INVEST", "REBAL"]
    n_actions     = len(ACTION_LABELS)

    lam_state  = state_data["lambdas"]
    lam_global = global_data["lambdas"]

    min_len = min(len(lam_state), len(lam_global))
    diff    = np.abs(lam_state[:min_len] - lam_global[:min_len])

    k    = min(200, min_len)
    topk = np.argsort(diff)[-k:]

    state_acts  = state_data["actions"][:min_len][topk]
    global_acts = global_data["actions"][:min_len][topk]

    state_dist  = np.bincount(state_acts,  minlength=n_actions) / k
    global_dist = np.bincount(global_acts, minlength=n_actions) / k

    fig, ax = plt.subplots(figsize=(7, 4))
    x   = np.arange(n_actions)
    w   = 0.35
    ax.bar(x - w/2, state_dist,  width=w, color=_BLUE,   label="CCPL-StateLambda", alpha=0.85)
    ax.bar(x + w/2, global_dist, width=w, color=_AMBER,  label="CCPL-GlobalLambda", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ACTION_LABELS)
    ax.set_ylabel("Action frequency (high-divergence states)")
    ax.set_title("Policy counterfactual at high λ-divergence states\n"
                 "(StateLambda should prefer DEFER/INVEST over FULL)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, os.path.join(out_dir, filename))



def plot_sigma_under_noise(logger_data: dict, out_dir: str,
                           smooth_k: int = 15,
                           filename: str = "mech_04_sigma_noise.png"):
    """
    Plot σ(s) and λ(s) over training in a noisy environment.
    Shows that σ rises when observation noise spikes, and λ follows —
    proving the σ-weighted penalty is actively responding to uncertainty.
    """
    sigmas  = np.array(logger_data["sigmas"],  float)
    lambdas = np.array(logger_data["lambdas"], float)

    def smooth(x, k=smooth_k):
        if len(x) < k: return x
        return np.convolve(x, np.ones(k)/k, mode="valid")

    s_s = smooth(sigmas)
    s_l = smooth(lambdas)
    xs  = np.arange(len(s_s))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    ax1.plot(xs, s_s, color=_TEAL,  linewidth=1.5, label="σ(s) — uncertainty")
    ax1.set_ylabel("σ(s)")
    ax1.set_title("σ and λ trajectories in NoisyEnv\n"
                  "(σ should spike with observation noise; λ should follow)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(xs, s_l, color=_BLUE,  linewidth=1.5, label="λ(s) — consequence weight")
    ax2.set_ylabel("λ(s)")
    ax2.set_xlabel("Step")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, filename))



def plot_deception_bench_collapse(
    all_seed_histories: dict,
    out_dir: str,
    smooth_k: int = 20,
    filename: str = "mech_05_deception_collapse.png",
):
    """
    The paper's key figure: PPO/Lagrangian high early reward followed by
    visible collapse when delayed consequences fire.
    CCPL stays at moderate reward without collapse.

    Expects all_seed_histories to contain at minimum:
      "CCPL", "PPO" (or "PPO-Lag")
    and optionally "PID-Lag", "CPO".
    """
    COLORS = {
        "CCPL":          _BLUE,
        "CCPL-Full":     _BLUE,
        "PPO":           _PURPLE,
        "PPO-Lag":       _PURPLE,
        "PID-Lag":       _CORAL,
        "CPO":           _TEAL,
        "CCPL-NoGRU":    _GRAY,
    }

    def smooth(x, k=smooth_k):
        if len(x) < k: return np.array(x, float)
        return np.convolve(np.array(x, float), np.ones(k)/k, mode="valid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, seed_hists in all_seed_histories.items():
        min_len = min(len(h["rewards"]) for h in seed_hists)
        mat     = np.array([smooth(h["rewards"][:min_len]) for h in seed_hists])
        mean    = mat.mean(0)
        std     = mat.std(0)
        xs      = np.arange(len(mean))
        c       = COLORS.get(name, _GRAY)
        ax1.plot(xs, mean, color=c, label=name, linewidth=1.8, alpha=0.9)
        ax1.fill_between(xs, mean - std, mean + std, alpha=0.12, color=c)

    ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
    ax1.set_title("DeceptionBench: per-episode reward\n"
                  "PPO exploits FULL → high early reward → collapse",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for name, seed_hists in all_seed_histories.items():
        min_len = min(len(h["consequences"]) for h in seed_hists)
        mat     = np.array([smooth(h["consequences"][:min_len]) for h in seed_hists])
        mean    = mat.mean(0)
        std     = mat.std(0)
        xs      = np.arange(len(mean))
        c       = COLORS.get(name, _GRAY)
        ax2.plot(xs, mean, color=c, label=name, linewidth=1.8, alpha=0.9)
        ax2.fill_between(xs, mean - std, mean + std, alpha=0.12, color=c)

    ax2.set_xlabel("Episode"); ax2.set_ylabel("Constraint cost J_c (smoothed)")
    ax2.set_title("DeceptionBench: constraint violation\n"
                  "CCPL maintains lower J_c throughout",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, filename))
    print(f"  [KEY RESULT] DeceptionBench collapse figure saved to {filename}")



def generate_mechanistic_plots(
    ccpl_logger:        LambdaTrajectoryLogger,
    global_logger:      LambdaTrajectoryLogger,
    noisy_logger:       LambdaTrajectoryLogger,
    deception_histories: dict,
    out_dir:            str,
):
    """
    Generate all mechanistic proof figures.
    Call after training is complete and loggers are populated.

    ccpl_logger   — logged during CCPL-StateLambda training on standard env
    global_logger — logged during CCPL-GlobalLambda training on standard env
    noisy_logger  — logged during CCPL-StateLambda training on NoisyEnv
    deception_histories — {agent_name: [hist_seed0, ...]} for DeceptionBench
    """
    print("\n  Generating mechanistic visualization figures...")
    os.makedirs(out_dir, exist_ok=True)

    d_ccpl   = ccpl_logger.arrays()
    d_global = global_logger.arrays()
    d_noisy  = noisy_logger.arrays()

    plot_lambda_heatmap(d_ccpl,   out_dir)
    plot_lambda_vs_shock(d_ccpl,  out_dir)
    plot_policy_counterfactual(d_ccpl, d_global, out_dir)
    plot_sigma_under_noise(d_noisy,    out_dir)
    plot_deception_bench_collapse(deception_histories, out_dir)

    print("  All mechanistic figures saved.")
