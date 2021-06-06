"""Returns features of this repository (e.g. version number)."""
from il_traffic.core.alg import DAgger
from il_traffic.core.env import ControllerEnv
from il_traffic.core.experts import IntelligentDriverModel
from il_traffic.core.experts import FollowerStopper
from il_traffic.core.experts import PISaturation
from il_traffic.core.experts import TimeHeadwayFollowerStopper
from il_traffic.core.model import FeedForwardModel
from .version import __version__ as v

# repo version number
__version__ = v

__all__ = [
    "DAgger",
    "ControllerEnv",
    "IntelligentDriverModel",
    "FollowerStopper",
    "PISaturation",
    "TimeHeadwayFollowerStopper",
    "FeedForwardModel",
    "__version__",
]
