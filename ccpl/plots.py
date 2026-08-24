import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "CCPL":         "#2563EB",
    "DQN":          "#DC2626",
    "DDQN":         "#D97706",
    "A2C":          "#16A34A",
    "PPO":          "#7C3AED",
    "CCPL-NoGRU":   "#0EA5E9",
    "CCPL-NoSigma": "#F97316",
    "CCPL-NoLambda":"#84CC16",
    "CCPL-NoMH":    "#EC4899",
}
SMOOTH_K = 20
LINE_KW  = dict(lw=1.8, alpha=0.9)


def _smooth(x, k=SMOOTH_K):
    if len(x) < k: return np.array(x, float)
    return np.convolve(np.array(x, float), np.ones(k)/k, mode="valid")


def _fig(title, xlabel, ylabel, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _find_ccpl_key(history: dict) -> str | None:
    if "CCPL" in history:
        return "CCPL"
    matches = [k for k in history if k.startswith("CCPL")]
    if matches:
        return matches[0]
    return None



def plot_ci_reward_curves(all_seed_histories, out):
    """all_seed_histories: {agent_name: [hist_seed0, hist_seed1, ...]}"""
    seed_counts = sorted({len(histories) for histories in all_seed_histories.values()})
    seed_label = str(seed_counts[0]) if len(seed_counts) == 1 else "varying"
    fig, ax = _fig(f"Per-Episode Reward — Mean ± 1 SD ({seed_label} seeds)",
                   "Episode", "Reward")
    for name, seed_hists in all_seed_histories.items():
        min_len = min(len(h["rewards"]) for h in seed_hists)
        mat     = np.array([_smooth(h["rewards"][:min_len]) for h in seed_hists])
        mean    = mat.mean(0); std = mat.std(0)
        x       = np.arange(len(mean))
        c       = COLORS.get(name, "#888")
        ax.plot(x, mean, color=c, label=name, **LINE_KW)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    ax.legend(fontsize=9); _save(fig, f"{out}/01_ci_reward_curves.png")



def plot_convergence_speed(all_seed_histories, out, threshold_pct=0.8):
    """Episodes to reach threshold_pct of max reward."""
    fig, ax = _fig("Convergence Speed (episodes to reach 80% peak reward)",
                   "Algorithm", "Episodes to Convergence")
    names, speeds = [], []
    for name, seed_hists in all_seed_histories.items():
        ep_list = []
        for h in seed_hists:
            r   = _smooth(h["rewards"])
            if len(r) == 0:
                ep_list.append(0)
                continue
            baseline = float(np.mean(r[:min(10, len(r))]))
            peak = float(r.max())
            cap = baseline + threshold_pct * (peak - baseline)
            hits = np.flatnonzero(r >= cap)
            ep_list.append(int(hits[0]) if peak > baseline + 1e-12 and len(hits)
                           else len(r))
        names.append(name); speeds.append(ep_list)

    means = [np.mean(s) for s in speeds]
    stds  = [np.std(s)  for s in speeds]
    order = np.argsort(means)
    names_  = [names[i] for i in order]
    means_  = [means[i] for i in order]
    stds_   = [stds[i]  for i in order]
    bars    = ax.bar(names_, means_, color=[COLORS.get(n,"#888") for n in names_],
                     yerr=stds_, capsize=4, width=0.55, alpha=0.85)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_ylabel("Episodes"); ax.grid(axis="y", alpha=0.3)
    _save(fig, f"{out}/02_convergence_speed.png")



def plot_cumulative_reward(histories, out):
    fig, ax = _fig("Cumulative Reward Over Training", "Episode", "Cumulative Reward")
    for n, h in histories.items():
        ax.plot(h["cumulative_reward"], color=COLORS.get(n,"#888"), label=n, **LINE_KW)
    ax.legend(fontsize=9); _save(fig, f"{out}/03_cumulative_reward.png")



def plot_delayed_consequence(histories, out):
    fig, ax = _fig("Delayed Consequence per Episode (smoothed)", "Episode", "Consequence")
    for n, h in histories.items():
        ax.plot(_smooth(h["consequences"]), color=COLORS.get(n,"#888"), label=n, **LINE_KW)
    ax.legend(fontsize=9); _save(fig, f"{out}/04_delayed_consequence.png")



def plot_stability(histories, out):
    fig, ax = _fig("Training Stability (rolling σ of reward)", "Episode", "Rolling Std")
    for n, h in histories.items():
        r = np.array(h["rewards"], float)
        if len(r) < SMOOTH_K: continue
        rs = [r[max(0,i-SMOOTH_K):i].std() for i in range(SMOOTH_K, len(r))]
        ax.plot(rs, color=COLORS.get(n,"#888"), label=n, **LINE_KW)
    ax.legend(fontsize=9); _save(fig, f"{out}/05_stability.png")



def plot_transfer_score(transfer_scores, out, title="Zero-Shot Transfer Score"):
    agents = list(transfer_scores.keys())
    scores = [transfer_scores[a] for a in agents]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars    = ax.bar(agents, scores, color=[COLORS.get(a,"#888") for a in agents],
                     width=0.5, alpha=0.88)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Transfer Score"); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=0.25)
    _save(fig, f"{out}/06_transfer_score.png")



def plot_compute_performance(histories, eval_results, env_names, out):
    fig, ax = _fig("Compute–Performance Tradeoff",
                   "Avg Inference Latency (ms)", "Mean Eval Reward")
    for name, h in histories.items():
        infer_ms = np.mean(h.get("infer_ms", [0.0]))
        mean_r   = np.mean([eval_results[name][e]["mean_reward"] for e in env_names
                            if name in eval_results and e in eval_results[name]])
        c = COLORS.get(name, "#888")
        ax.scatter(infer_ms, mean_r, color=c, s=120, zorder=5)
        ax.annotate(name, (infer_ms, mean_r), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color=c)
    ax.grid(alpha=0.25, linestyle="--")
    _save(fig, f"{out}/07_compute_performance.png")



def plot_per_env_ranking(eval_results, env_names, out):
    agents = list(eval_results.keys())
    x      = np.arange(len(env_names))
    width  = 0.9 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, agent in enumerate(agents):
        if agent not in eval_results: continue
        vals = []
        errs = []
        for e in env_names:
            if e in eval_results[agent]:
                vals.append(eval_results[agent][e]["mean_reward"])
                errs.append(eval_results[agent][e].get("std_reward", 0.0))
            else:
                vals.append(0); errs.append(0)
        ax.bar(x + i*width, vals, width, label=agent,
               yerr=errs, capsize=3,
               color=COLORS.get(agent,"#888"), alpha=0.85)

    ax.set_xticks(x + width*(len(agents)-1)/2)
    ax.set_xticklabels(env_names, rotation=15)
    ax.set_ylabel("Mean Reward"); ax.set_title("Per-Environment Ranking", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    _save(fig, f"{out}/08_per_env_ranking.png")



def plot_ablation_comparison(ablation_results, env_names, out):
    """ablation_results: {variant_name: {env: metrics}}"""
    if not ablation_results:
        return
    agents  = list(ablation_results.keys())
    means   = [np.mean([ablation_results[a][e]["mean_reward"] for e in env_names
                        if e in ablation_results[a]]) for a in agents]
    cons    = [np.mean([ablation_results[a][e]["mean_consequence"] for e in env_names
                        if e in ablation_results[a]]) for a in agents]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for vals, ylabel, title, ax in [
        (means, "Mean Reward",      "Ablation: Reward Impact",      axes[0]),
        (cons,  "Mean Consequence", "Ablation: Consequence Impact",  axes[1]),
    ]:
        bars = ax.bar(agents, vals, color=[COLORS.get(a,"#888") for a in agents],
                      width=0.55, alpha=0.88)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    _save(fig, f"{out}/09_ablation_comparison.png")



def plot_unseen_transfer(unseen_results, unseen_envs, out):
    if not unseen_results:
        return
    agents = list(unseen_results.keys())
    x      = np.arange(len(unseen_envs))
    width  = 0.9 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, agent in enumerate(agents):
        if agent not in unseen_results: continue
        vals = [unseen_results[agent].get(e, {}).get("mean_reward", 0) for e in unseen_envs]
        errs = [unseen_results[agent].get(e, {}).get("std_reward",  0) for e in unseen_envs]
        ax.bar(x + i*width, vals, width, label=agent, yerr=errs, capsize=3,
               color=COLORS.get(agent,"#888"), alpha=0.85)

    ax.set_xticks(x + width*(len(agents)-1)/2)
    ax.set_xticklabels(unseen_envs, rotation=10)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Zero-Shot Generalization — Unseen Environments",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    _save(fig, f"{out}/10_unseen_transfer.png")



def plot_sample_efficiency(histories, out):
    fig, ax = _fig("Running Mean Training Return", "Episode", "Mean Return")
    for n, h in histories.items():
        eff = np.array(h["cumulative_reward"]) / (np.arange(1, len(h["rewards"])+1))
        ax.plot(_smooth(eff), color=COLORS.get(n,"#888"), label=n, **LINE_KW)
    ax.legend(fontsize=9); _save(fig, f"{out}/11_sample_efficiency.png")



def plot_regret_reduction(histories, out, use_per_agent_oracle: bool = False):
    if use_per_agent_oracle:
        title  = "Cumulative Gap to Each Agent's Best Observed Episode"
        get_best = lambda name, h: max(h["rewards"])
    else:
        title  = "Cumulative Gap to the Best Observed Training Episode"
        global_best = max(max(h["rewards"]) for h in histories.values())
        get_best = lambda name, h: global_best

    fig, ax = _fig(title, "Episode", "Cumulative observed-return gap")
    for n, h in histories.items():
        best   = get_best(n, h)
        regret = np.cumsum([best - r for r in h["rewards"]])
        ax.plot(regret, color=COLORS.get(n,"#888"), label=n, **LINE_KW)
    ax.legend(fontsize=9); _save(fig, f"{out}/12_regret_reduction.png")



def plot_final_ranking(eval_results, env_names, out):
    agents = [n for n in eval_results if n in eval_results]
    means  = [np.mean([eval_results[a][e]["mean_reward"] for e in env_names
                       if e in eval_results[a]]) for a in agents]
    order  = np.argsort(means)[::-1]
    as_    = [agents[i] for i in order]
    ms_    = [means[i]  for i in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars    = ax.bar(as_, ms_, color=[COLORS.get(a,"#888") for a in as_], alpha=0.88)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Final Benchmark Ranking", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, f"{out}/13_final_ranking.png")



def plot_ccpl_diagnostics(history, out):
    ccpl_key = _find_ccpl_key(history)
    if ccpl_key is None:
        warnings.warn(
            "plot_ccpl_diagnostics: no key starting with 'CCPL' found in histories dict. "
            "Skipping CCPL diagnostics plot.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    h   = history[ccpl_key]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r_s = _smooth(h["rewards"])
    c_s = _smooth(h["consequences"])
    n   = min(len(r_s), len(c_s))
    sc  = axes[0].scatter(c_s[:n], r_s[:n], c=np.arange(n), cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label="Episode")
    axes[0].set_xlabel("Consequence"); axes[0].set_ylabel("Reward")
    axes[0].set_title(f"{ccpl_key}: Reward vs Consequence Trade-off", fontweight="bold")
    axes[0].grid(alpha=0.25)

    axes[1].plot(_smooth(h["delayed_hits"]), color=COLORS.get(ccpl_key, COLORS["CCPL"]), **LINE_KW)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Delayed Hits (smoothed)")
    axes[1].set_title(f"{ccpl_key}: Delayed Hits Over Training", fontweight="bold")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = f"{out}/14_ccpl_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")



def plot_param_count(histories, out):
    names  = [n for n, h in histories.items() if "param_count" in h]
    counts = [histories[n]["param_count"] for n in names]
    if not names: return
    fig, ax = plt.subplots(figsize=(9, 5))
    bars    = ax.bar(names, counts, color=[COLORS.get(n,"#888") for n in names], alpha=0.88)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_ylabel("Parameter Count")
    ax.set_title("Model Parameter Count", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, f"{out}/15_param_count.png")


def generate_all_plots(histories, eval_results, transfer_scores, eval_envs, out_dir,
                       all_seed_histories=None, ablation_results=None,
                       unseen_results=None, unseen_envs=None,
                       regret_per_agent_oracle: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    print("\nGenerating plots...")

    if all_seed_histories:
        plot_ci_reward_curves(all_seed_histories, out_dir)
        plot_convergence_speed(all_seed_histories, out_dir)

    plot_cumulative_reward(histories, out_dir)
    plot_delayed_consequence(histories, out_dir)
    plot_stability(histories, out_dir)
    plot_transfer_score(transfer_scores, out_dir)
    plot_compute_performance(histories, eval_results, eval_envs, out_dir)
    plot_per_env_ranking(eval_results, eval_envs, out_dir)

    if ablation_results:
        plot_ablation_comparison(ablation_results, eval_envs, out_dir)

    if unseen_results and unseen_envs:
        plot_unseen_transfer(unseen_results, unseen_envs, out_dir)

    plot_sample_efficiency(histories, out_dir)
    plot_regret_reduction(histories, out_dir,
                          use_per_agent_oracle=regret_per_agent_oracle)
    plot_final_ranking(eval_results, eval_envs, out_dir)
    plot_ccpl_diagnostics(histories, out_dir)
    plot_param_count(histories, out_dir)

    print(f"All plots saved -> {out_dir}/")
