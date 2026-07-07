"""Model zoo. Importing this package registers all built-in detectors.

`build_model(cfg)` constructs whatever `cfg.name`/`cfg.arch` names — the trainer,
evaluator, inference, and export code are all model-agnostic through this factory.
"""

from . import baselines  # noqa: F401  (registers lfcc_lcnn, rawnet2, aasist_raw)
from .auralguard import AuralGuard  # noqa: F401  (registers auralguard)
from .registry import available, build_model, register

__all__ = ["AuralGuard", "build_model", "register", "available"]
