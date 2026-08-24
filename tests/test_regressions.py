import importlib
import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from a2c_agent import A2CAgent
from adversarial_envs import SafetyGymAdapter
from causal_consequence import CausalHistoryEncoder, CausalReplayBuffer
from ccpl_agent import make_ccpl, run_episode
from constrained_baselines import (
    CPOAgent,
    PIDLagrangianAgent,
    RCPOAgent,
    SACLagrangianAgent,
)
from delay_bellman import DelayCorrectedBellman, DelayDistributionNet
from dqn_agent import DQNAgent
from environments import (
    AdversarialEnv,
    DeceptiveRewardEnv,
    NoisyEnv,
    RandomisedEnv,
    ResourceCollapseEnv,
    ShiftedConsequenceEnv,
    StandardEnv,
)
from networks import (
    Adam,
    CriticNetwork,
    LayerNorm,
    Linear,
    MultiHorizonConsequenceNet,
    QNetwork,
    gru_batch_forward,
    softmax,
)
from networks_v7 import AttentionAugmentedPolicyNet, sinusoidal_pe
from normalizer import StateNormalizer
from ppo_agent import PPOAgent
from stats import full_comparison_table, mannwhitney, paired_randomization
from train import count_parameters, run_episode_baseline


def test_softmax_is_stable_for_large_logits():
    probabilities = softmax(np.array([[1000.0, 999.0, -1000.0]], np.float32))
    assert np.all(np.isfinite(probabilities))
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-7)
    assert probabilities[0, 0] > probabilities[0, 1] > probabilities[0, 2]


def test_layernorm_backward_matches_finite_difference():
    rng = np.random.default_rng(2)
    layer = LayerNorm(4, eps=1e-5)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    analytical, dg, db = layer.backward(upstream)

    epsilon = 1e-3
    numerical = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        f_plus = float(np.sum(layer.forward(plus) * upstream))
        f_minus = float(np.sum(layer.forward(minus) * upstream))
        numerical[index] = (f_plus - f_minus) / (2.0 * epsilon)

    assert analytical.shape == x.shape
    assert dg.shape == (4,)
    assert db.shape == (4,)
    assert np.max(np.abs(analytical - numerical)) < 3e-3


def test_linear_backward_supports_sequence_tensors():
    rng = np.random.default_rng(3)
    layer = Linear(4, 2, rng)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=(2, 3, 2)).astype(np.float32)
    layer.forward(x)
    dx, d_weight, d_bias = layer.backward(upstream)
    assert dx.shape == x.shape
    assert np.allclose(d_weight, x.reshape(-1, 4).T @ upstream.reshape(-1, 2))
    assert np.allclose(d_bias, upstream.reshape(-1, 2).sum(axis=0))


def test_value_and_q_updates_move_predictions_toward_targets():
    state = np.array([[0.3, -0.2, 0.7]], np.float32)
    critic = CriticNetwork(3, hidden_dim=8, n_layers=1, lr=1e-2, seed=4)
    before = float(critic.value(state)[0])
    critic.backward_update(state, np.array([1.0], np.float32), np.ones(1))
    after = float(critic.value(state)[0])
    assert after > before

    q_net = QNetwork(3, 3, hidden_dim=8, n_layers=1, lr=1e-2, seed=5)
    before_q = float(q_net.forward(state)[0, 1])
    q_net.backward_update(
        state, np.array([1]), np.array([1.0], np.float32), np.ones(1))
    after_q = float(q_net.forward(state)[0, 1])
    assert after_q > before_q


def test_adam_rejects_missing_or_malformed_gradients():
    parameter = np.zeros((2, 2), np.float32)
    optimizer = Adam([parameter])
    with pytest.raises(ValueError):
        optimizer.step([])
    with pytest.raises(ValueError):
        optimizer.step([np.zeros(2, np.float32)])
    with pytest.raises(FloatingPointError):
        optimizer.step([np.full((2, 2), np.nan, np.float32)])


def test_normalizer_does_not_saturate_after_one_sample():
    normalizer = StateNormalizer(2)
    sample = np.array([0.4, 0.7], np.float32)
    normalizer.update(sample)
    assert np.allclose(normalizer.normalize(sample), sample)
    normalizer.update(np.array([0.5, 0.6], np.float32))
    assert np.all(np.isfinite(normalizer.normalize(sample)))


def test_history_encoder_is_deterministic_and_has_no_untrained_parameters():
    encoder_a = CausalHistoryEncoder(3, 2, hidden_dim=8, seed=1)
    encoder_b = CausalHistoryEncoder(3, 2, hidden_dim=8, seed=999)
    history = [
        (np.array([0.1, 0.2, 0.3], np.float32), 0, 0.0),
        (np.array([0.4, 0.5, 0.6], np.float32), 1, 1.0),
    ]
    assert np.allclose(
        encoder_a.context_batch([history]), encoder_b.context_batch([history]))
    assert encoder_a.params() == []


def test_causal_replay_keeps_observed_and_source_aligned_costs_separate():
    buffer = CausalReplayBuffer(
        capacity=8, history_len=2, state_dim=2, action_dim=2, gru_dim=3, seed=7)
    first = buffer.push(
        [0.1, 0.2], 0, 1.0, [0.2, 0.3], 0.0, False,
        np.zeros(3), np.zeros(3), [], scm_label_valid=False)
    buffer.push(
        [0.2, 0.3], 1, 0.5, [0.3, 0.4], 4.0, True,
        np.zeros(3), np.zeros(3), [], scm_label_valid=True)
    assert buffer.set_aligned_consequence(first, 4.0)
    batch = buffer.sample(2)
    by_action = {int(action): i for i, action in enumerate(batch["actions"])}
    assert batch["consequences"][by_action[1]] == pytest.approx(4.0)
    assert batch["aligned_consequences"][by_action[0]] == pytest.approx(4.0)
    assert bool(batch["aligned_valid"][by_action[0]])
    assert bool(batch["scm_label_valid"][by_action[1]])


@pytest.mark.parametrize("env_cls", [
    StandardEnv,
    NoisyEnv,
    ShiftedConsequenceEnv,
    RandomisedEnv,
    AdversarialEnv,
    DeceptiveRewardEnv,
    ResourceCollapseEnv,
])
def test_environment_returned_totals_match_episode_stats(env_cls):
    env = env_cls(max_steps=12, consequence_delay=2, seed=11)
    env.reset()
    rng = np.random.default_rng(12)
    reward_sum = cost_sum = 0.0
    while not env.done:
        _, reward, cost, _, info = env.step(int(rng.integers(env.action_dim)))
        reward_sum += reward
        cost_sum += cost
        assert info["total_reward"] == pytest.approx(reward_sum)
        assert info["total_consequence"] == pytest.approx(cost_sum)
    stats = env.episode_stats()
    assert stats["total_reward"] == pytest.approx(reward_sum)
    assert stats["total_consequence"] == pytest.approx(cost_sum)


def test_shifted_cost_is_modified_before_delay_queue():
    env = ShiftedConsequenceEnv(max_steps=12, seed=19)
    env.noise_std = 0.0
    env.reset()
    immediate = []
    emitted = []
    for step in range(11):
        action = 2 if step == 0 else 0
        _, _, cost, _, info = env.step(action)
        immediate.append(info["immediate_consequence"])
        emitted.append(cost)
    assert emitted[10] == pytest.approx(immediate[0])


def test_only_matching_standard_environment_enables_scm_labels():
    standard = StandardEnv(max_steps=2, seed=1)
    standard.reset()
    *_, standard_info = standard.step(0)
    noisy = NoisyEnv(max_steps=2, seed=1)
    noisy.reset()
    *_, noisy_info = noisy.step(0)
    assert standard_info["scm_label_valid"] is True
    assert noisy_info["scm_label_valid"] is False


def test_delay_distribution_includes_zero_and_keeps_reward_modulus_gamma():
    network = DelayDistributionNet(gru_dim=4, hidden_dim=8, tau_max=3, seed=2)
    hidden = np.zeros((5, 4), np.float32)
    probabilities = network.forward(hidden)
    assert probabilities.shape == (5, 4)
    network.update_step(hidden, np.zeros(5, np.int32), np.ones(5, np.float32))
    bellman = DelayCorrectedBellman(network, gamma=0.9, tau_max=3)
    diagnostic = bellman.verify_contraction(hidden)
    assert diagnostic["contraction_modulus"] == pytest.approx(0.9)
    assert 0.9 ** 3 <= diagnostic["gamma_eff_min"] <= 1.0
    assert 0.9 ** 3 <= diagnostic["gamma_eff_max"] <= 1.0


def test_delay_observation_uses_current_hidden_state():
    agent = make_ccpl(
        3, 2, seed=32, pretrain_steps=0, batch_size=2,
        buffer_capacity=8, gru_dim=4, hidden_dim=8, n_layers=1,
        tau_max=1)
    state = np.array([0.1, 0.2, 0.3], np.float32)
    next_state = np.array([0.2, 0.3, 0.4], np.float32)
    hidden = np.zeros((1, 4), np.float32)
    next_hidden = np.ones((1, 4), np.float32)

    agent.store(
        state, 0, 0.0, next_state, 0.0, False,
        hidden=hidden, next_hidden=next_hidden,
        info={"actual_tau": 0, "delay_supervision_valid": True,
              "scm_label_valid": False})

    assert np.allclose(agent._obs_tau_buffer[-1][0], next_hidden.squeeze())


def test_ccpl_update_conditions_delay_discount_on_current_hidden_state():
    agent = make_ccpl(
        3, 2, seed=33, pretrain_steps=0, batch_size=4,
        buffer_capacity=16, gru_dim=4, hidden_dim=8, n_layers=1,
        tau_max=2)
    states = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.3, 0.2, 0.1],
        [0.4, 0.3, 0.2],
    ], np.float32)
    hiddens = np.zeros((4, 4), np.float32)
    batch = {
        "states": states,
        "actions": np.array([0, 1, 0, 1], np.int32),
        "rewards": np.zeros(4, np.float32),
        "next_states": states + 0.05,
        "consequences": np.zeros(4, np.float32),
        "aligned_consequences": np.zeros(4, np.float32),
        "aligned_valid": np.zeros(4, bool),
        "scm_label_valid": np.zeros(4, bool),
        "dones": np.zeros(4, np.float32),
        "hiddens": hiddens,
        "next_hiddens": np.zeros((4, 4), np.float32),
        "weights": np.ones(4, np.float32),
        "indices": np.arange(4),
        "histories": [[] for _ in range(4)],
    }
    expected_current = gru_batch_forward(agent.policy_net.gru, states, hiddens)
    captured = {}
    gamma_inputs = []

    agent.buffer.sample = lambda batch_size: batch
    agent.buffer.update_priorities = lambda *args, **kwargs: None

    def fake_delay_update(h_batch, observed_tau, weights):
        captured["delay_h"] = np.asarray(h_batch, np.float32).copy()
        return 0.0

    def fake_gamma_eff(h_batch):
        gamma_inputs.append(np.asarray(h_batch, np.float32).copy())
        return np.full(len(h_batch), 0.75, np.float32)

    agent.delay_dist.update_step = fake_delay_update
    agent.bellman.gamma_eff = fake_gamma_eff

    assert agent.update()
    assert np.allclose(captured["delay_h"], expected_current)
    assert np.allclose(gamma_inputs[-1], expected_current)


def test_causal_replay_full_batch_importance_weights_are_uniform():
    buffer = CausalReplayBuffer(
        capacity=4, history_len=2, state_dim=2, action_dim=2, gru_dim=3,
        seed=12)
    for i in range(4):
        value = np.full(2, i, np.float32)
        buffer.push(
            value, i % 2, 0.0, value + 1.0, 0.0, False,
            np.zeros(3), np.zeros(3), [], scm_label_valid=False)
    buffer.update_priorities(
        np.arange(4), np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([8.0, 4.0, 2.0, 1.0]))

    batch = buffer.sample(4)

    assert set(batch["indices"]) == {0, 1, 2, 3}
    assert np.allclose(batch["weights"], np.ones(4, np.float32))


def test_multihorizon_consequence_loss_decreases():
    model = MultiHorizonConsequenceNet(
        state_dim=3, action_dim=2, hidden_dim=8, n_layers=1, lr=2e-3, seed=8)
    states = np.tile(np.array([[0.2, 0.5, 0.8]], np.float32), (8, 1))
    actions = np.arange(8, dtype=np.int32) % 2
    targets = np.full(8, 0.25, np.float32)
    weights = np.ones(8, np.float32)
    initial = float(np.mean((model.predict(states, actions)[0] - targets) ** 2))
    for _ in range(25):
        model.update_step(states, actions, targets, weights)
    final = float(np.mean((model.predict(states, actions)[0] - targets) ** 2))
    assert final < initial


def test_attention_memory_parameters_receive_gradients():
    network = AttentionAugmentedPolicyNet(
        state_dim=3, action_dim=2, gru_dim=4, hidden_dim=8, n_layers=1,
        d_model=4, n_heads=2, memory_capacity=4, seed=9)
    memory_states = np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]], np.float32)
    network.memory.write(memory_states, np.array([1.0, 2.0], np.float32))
    states = np.array([[0.1, 0.4, 0.2], [0.7, 0.2, 0.9]], np.float32)
    hidden = network.zero_state(2)
    before = network.memory.query_proj.W.copy()
    network.backward_update(
        states, hidden, np.array([0, 1]),
        np.array([1.0, -0.5], np.float32), np.ones(2, np.float32))
    assert not np.allclose(network.memory.query_proj.W, before)
    assert all(np.all(np.isfinite(parameter)) for parameter in network.memory.params())


def test_sinusoidal_positional_encoding_supports_odd_dimensions():
    encoding = sinusoidal_pe(seq_len=4, d_model=5)
    assert encoding.shape == (4, 5)
    assert np.all(np.isfinite(encoding))
    assert np.allclose(encoding[0, 0::2], 0.0)
    assert np.allclose(encoding[0, 1::2], 1.0)


def test_ccpl_short_run_updates_and_aligns_delayed_feedback():
    agent = make_ccpl(
        6, 5, seed=21, pretrain_steps=0, batch_size=4,
        buffer_capacity=64, gru_dim=8, hidden_dim=16, n_layers=1,
        tau_max=4)
    result = run_episode(
        agent, StandardEnv(max_steps=10, consequence_delay=2, seed=22), train=True,
        update_freq=1)
    assert result["steps"] > 0
    assert agent.update_count > 0
    assert np.all(np.isfinite(list(agent.policy_net._flat()[0].ravel())))
    assert np.count_nonzero(agent.buffer._aligned_valid[:len(agent.buffer)]) > 0

    run_episode(
        agent, StandardEnv(max_steps=5, consequence_delay=1, seed=23), train=False)
    assert len(agent.icn.encoder._history) > 0


def test_small_baseline_training_runs_are_finite():
    common = dict(state_dim=6, action_dim=5, hidden_dim=8, n_layers=1, seed=31)
    agents = [
        DQNAgent(**common, batch_size=4, buffer_capacity=32),
        PPOAgent(**common, n_steps=4, n_epochs=1, mini_batch_size=4),
        A2CAgent(**common, n_steps=4),
        CPOAgent(**common, n_steps=4),
        RCPOAgent(**common, n_steps=4, n_epochs=1, mini_batch_size=4),
        PIDLagrangianAgent(
            **common, n_steps=4, n_epochs=1, mini_batch_size=4),
        SACLagrangianAgent(**common, batch_size=4, buffer_capacity=32),
    ]
    for index, agent in enumerate(agents):
        result = None
        for attempt in range(3):
            result = run_episode_baseline(
                agent, StandardEnv(
                    max_steps=8, consequence_delay=1,
                    seed=40 + index + 100 * attempt),
                train=True, update_freq=1)
            if agent.update_count:
                break
        assert result["steps"] > 0
        assert agent.update_count > 0
        for parameter_count in [count_parameters(agent)]:
            assert parameter_count > 0


def test_parameter_counter_excludes_dqn_target_network():
    agent = DQNAgent(3, 2, hidden_dim=8, n_layers=1, seed=2)
    expected = sum(parameter.size for parameter in agent.online.optim.params)
    assert count_parameters(agent) == expected


def test_seed_paired_randomization_and_descriptive_guard():
    test = paired_randomization(
        [5, 6, 7, 8, 9], [1, 2, 3, 4, 5], alternative="greater")
    assert test["valid"]
    assert test["exact"]
    assert test["p"] == pytest.approx(1 / 32)

    cost_test = paired_randomization(
        [1, 2, 3, 4, 5], [5, 6, 7, 8, 9], alternative="less")
    assert cost_test["r"] > 0

    episode_only = {
        "CCPL": {"env": {"rewards": [1.0, 2.0]}},
        "Base": {"env": {"rewards": [0.0, 1.0]}},
    }
    row = full_comparison_table(episode_only, ["env"])[0]
    assert not row["inferential_valid"]
    assert row["p"] is None


def test_two_sided_mannwhitney_preserves_effect_direction():
    pytest.importorskip("scipy")
    result = mannwhitney([5, 6, 7], [1, 2, 3], alternative="two-sided")
    assert result["valid"]
    assert result["r"] > 0


class _FakeBox:
    shape = (2,)
    low = np.array([-1.0, -2.0], np.float32)
    high = np.array([1.0, 2.0], np.float32)


class _FakeSafetyRaw:
    action_space = _FakeBox()

    def step(self, action):
        self.last_action = np.asarray(action)
        return np.ones(3), 2.0, 1.5, False, False, {"source": "fake"}


def test_official_safety_adapter_handles_six_value_api_and_axis_actions():
    adapter = SafetyGymAdapter.__new__(SafetyGymAdapter)
    adapter._raw = _FakeSafetyRaw()
    adapter.action_wrapper = "axis"
    adapter.raw_action_dim = 2
    adapter.action_dim = 5
    adapter.max_steps = 10
    adapter._step = 0
    adapter._done = False
    adapter._cum_cost = 0.0
    adapter._cum_reward = 0.0
    adapter._delayed_hits = 0
    mapped = adapter._map_action(4)
    assert np.allclose(mapped, [0.0, -2.0])
    _, reward, cost, done, info = adapter.step(1)
    assert reward == pytest.approx(2.0)
    assert cost == pytest.approx(1.5)
    assert done is False
    assert info["actual_tau"] == 0
    assert info["delayed_hits"] == 1


def test_generate_plots_import_has_no_execution_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("generate_plots")
    importlib.reload(module)
    assert list(tmp_path.iterdir()) == []
