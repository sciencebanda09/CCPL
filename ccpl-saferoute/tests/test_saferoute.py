from saferoute.env import SafeRouteEnv


def test_reset_and_step_shapes():
    env = SafeRouteEnv(seed=1, delay=3)
    state = env.reset()
    next_state, reward, cost, done, info = env.step(3)
    assert state.shape == (12,)
    assert next_state.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(cost, float)
    assert isinstance(done, bool)
    assert info["delay"] == 3


def test_delayed_cost_is_emitted_after_configured_delay():
    env = SafeRouteEnv(size=8, delay=2, seed=1)
    env.reset()
    env.position = (3, 3)
    env.step(3)
    _, _, first_cost, _, _ = env.step(4)
    _, _, second_cost, _, _ = env.step(4)
    assert first_cost == 0.0
    assert second_cost > 0.0


def test_render_contains_start_goal_and_hazard():
    env = SafeRouteEnv(size=8, delay=1, seed=2)
    board = env.render_text()
    assert "R" in board
    assert "G" in board
    assert "X" in board


def test_cost_budget_and_causal_delta_are_reported():
    env = SafeRouteEnv(size=8, delay=0, seed=3)
    env.reset()
    env.position = (3, 3)
    _, _, cost, _, info = env.step(3)
    assert cost == info["cost"]
    assert info["causal_delta"] >= 0.0
    assert info["budget"] == 3.0


def test_seeded_resets_generate_recorded_layouts():
    env = SafeRouteEnv(size=8, delay=1, seed=4)
    env.reset(seed=10)
    first = set(env.hazards)
    env.reset(seed=11)
    second = set(env.hazards)
    assert first != second
    assert (1, 1) not in second
    assert env.goal not in second
