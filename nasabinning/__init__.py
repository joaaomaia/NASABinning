# Init file
"""
NASABinning: binning auditável com foco em estabilidade e separação temporal das curvas.

A biblioteca expõe a classe principal `NASABinner`.
"""
from .binning_engine import NASABinner

__all__ = ["NASABinner"]
__version__ = "0.1.0"
