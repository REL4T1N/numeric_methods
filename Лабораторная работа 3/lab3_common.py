"""Shared checks and error plots for laboratory work 3."""
from pathlib import Path
import math
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def check_nodes(xs, ys, minimum=2):
    if len(xs) != len(ys) or len(xs) < minimum:
        raise ValueError(f"Need at least {minimum} paired nodes")
    if any(not math.isfinite(v) for v in (*xs, *ys)):
        raise ValueError("All table entries must be finite")
    if any(right <= left for left, right in zip(xs, xs[1:])):
        raise ValueError("Nodes must be strictly increasing")


def plot_errors(counts, errors, filename, xlabel="Число узлов / разбиений"):
    """Plot a meaningful refinement sequence; errors must share one reference."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in errors.items():
        if len(values) != len(counts):
            raise ValueError("Counts and errors have different lengths")
        ax.semilogy(counts, [max(abs(v), 1e-16) for v in values], "o-", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Абсолютная погрешность")
    ax.set_xticks(counts)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=150)
    plt.close(fig)
