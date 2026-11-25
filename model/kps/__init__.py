"""Karplus--Strong string models: on the integer (L, p) grid, and relaxed to continuous delays."""

from .pluck import KarplusStrongPluck, KarplusStrongPluckLagrange
from .triangle import KarplusStrongTriangle, KarplusStrongTriangleLagrange

# Excitation name -> string model.
INTEGER_MODELS = {"comb": KarplusStrongPluck, "triangle": KarplusStrongTriangle}
LAGRANGE_MODELS = {"comb": KarplusStrongPluckLagrange, "triangle": KarplusStrongTriangleLagrange}
