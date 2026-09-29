"""Matplotlib, imported only when a figure is drawn.

The metric functions in this package are also used by the service, whose runtime image
leaves out the `train` dependency group and therefore Matplotlib. The service builds
its monitoring report from `compute_metrics` and `calibration_metrics`, so importing the
module that holds them must not import a plotting library the image does not have.
"""

from __future__ import annotations

from typing import Any


def pyplot() -> Any:
    """`matplotlib.pyplot` on the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
