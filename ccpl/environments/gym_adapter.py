"""Adapter from Gymnasium environments to the CCPL episode interface."""


class GymnasiumCCPLEnv:
    """Expose a Gymnasium environment with a separate consequence signal.

    The wrapped environment must place the safety cost in ``info`` under
    ``consequence_key`` (``"cost"`` by default). Legacy four-value ``step``
    results are accepted as well.
    """

    def __init__(self, env, consequence_key: str = "cost", consequence_delay: int = 0):
        if not hasattr(env, "reset") or not hasattr(env, "step"):
            raise TypeError("env must provide reset() and step() methods")
        if int(consequence_delay) < 0:
            raise ValueError("consequence_delay must be non-negative")
        self.env = env
        self.consequence_key = str(consequence_key)
        self.consequence_delay = int(consequence_delay)
        self.done = False
        self._steps = 0
        self._delayed_hits = 0
        self._total_consequence = 0.0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        observation = result[0] if isinstance(result, tuple) else result
        self.done = False
        self._steps = 0
        self._delayed_hits = 0
        self._total_consequence = 0.0
        return observation

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            observation, reward, done, info = result
            done = bool(done)
        else:
            raise ValueError("wrapped env.step() must return 4 or 5 values")
        info = dict(info or {})
        consequence = float(info.get(self.consequence_key, 0.0))
        self.done = done
        self._steps += 1
        self._total_consequence += consequence
        self._delayed_hits += int(consequence > 0.0)
        return observation, float(reward), consequence, done, info

    def episode_stats(self):
        return {
            "steps": self._steps,
            "delayed_hits": self._delayed_hits,
            "total_consequence": self._total_consequence,
        }
