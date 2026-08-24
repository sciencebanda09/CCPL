"""Public CCPL reinforcement-learning API.

The lower-level modules remain available for research extensions, but typical
users can import the agent and environment factories directly from ``ccpl``.
"""

from .algorithms.ccpl_agent import CCPLAgent, make_ccpl, make_ccpl_base, run_episode
from .environments.environments import ENV_REGISTRY, make_env
from .environments.gym_adapter import GymnasiumCCPLEnv
from .safety import SafetyPolicy, SafetyTrip

__version__ = "0.7.0"

__all__ = [
    "CCPLAgent",
    "ENV_REGISTRY",
    "GymnasiumCCPLEnv",
    "SafetyPolicy",
    "SafetyTrip",
    "make_ccpl",
    "make_ccpl_base",
    "make_env",
    "run_episode",
]
