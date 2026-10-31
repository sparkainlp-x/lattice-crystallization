"""lattice-crystallization: lateral score on streaming-PCA patterns.

SYNTHETIC classical toy only. No biology, no medicine, no quantum
hardware, no brain or consciousness claim.
"""

from .experiment import run_experiment
from .gate import calibrate_gate, overlap
from .lat import combined_score, lateral_score
from .lattice import build_pattern, row_slip, seam_shift
from .stream import crystallize_family

__version__ = "0.1.0"
__all__ = ["run_experiment", "calibrate_gate", "overlap",
           "combined_score", "lateral_score", "build_pattern",
           "row_slip", "seam_shift", "crystallize_family", "__version__"]
