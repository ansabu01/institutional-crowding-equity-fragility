"""Shared graph settings for thesis notebooks."""

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


COLOR_PALETTE = [
    "#377EB8",
    "#E41A1C",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
]

SINGLE_PANEL_SIZE = (9, 4.5)
TWO_PANEL_SIZE = (9, 6)

_GRID_COLOR = "#D9D9D9"


def set_matplotlib_style() -> None:
    """Apply the shared Matplotlib thesis style."""

    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",

            # Axes
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,

            # Keep only left and bottom axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,

            # Grid
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": _GRID_COLOR,
            "grid.alpha": 1.0,
            "grid.linewidth": 0.6,

            # Font / text
            "font.size": 15,
            "text.color": "black",
            "axes.labelcolor": "black",
            "font.family": "DejaVu Sans",

            # Title
            "axes.titlesize": 15,
            "axes.titleweight": "normal",
            "axes.titlecolor": "black",
            "axes.titlepad": 6,

            # Axis titles
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "axes.labelpad": 12,

            # Tick labels
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.color": "black",
            "ytick.color": "black",

            # Tick marks
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,

            # Lines
            "lines.linewidth": 1.8,

            # Colors
            "axes.prop_cycle": cycler(color=COLOR_PALETTE),

            # Legend
            "legend.frameon": False,
            "legend.fontsize": 12,
            "legend.labelcolor": "black",
        }
    )


def apply_matplotlib_layout(figure: Figure) -> None:
    """Apply shared spacing to a completed Matplotlib figure."""

    figure.tight_layout(h_pad=2.0)


def thesis_plotnine_theme(
    figure_size: tuple[float, float] = SINGLE_PANEL_SIZE,
    *,
    show_legend: bool = True,
):
    """Return the shared plotnine theme."""
    from plotnine import (
        element_blank,
        element_line,
        element_text,
        theme,
        theme_minimal,
    )

    plot_theme = theme_minimal(base_size=10) + theme(
        figure_size=figure_size,
        panel_spacing_y=0.04,
        plot_margin=0.025,
        panel_grid_major=element_line(color=_GRID_COLOR, size=0.6),
        panel_grid_minor=element_blank(),
        axis_line_x=element_line(size=0.8),
        axis_line_y=element_line(size=0.8),
        axis_ticks_major=element_line(color="black", size=0.8),
        axis_ticks_length=4,
        text=element_text(color="black"),
        plot_title=element_text(size=12, margin={"b": 10}),
        axis_title=element_blank(),
        axis_text_x=element_text(size=9, margin={"t": 6}),
        axis_text_y=element_text(size=9, margin={"r": 6}),
        strip_text_y=element_text(size=10),
        legend_title=element_blank(),
        legend_text=element_text(size=9),
    )

    if not show_legend:
        plot_theme += theme(legend_position="none")

    return plot_theme


__all__ = [
    "COLOR_PALETTE",
    "SINGLE_PANEL_SIZE",
    "TWO_PANEL_SIZE",
    "apply_matplotlib_layout",
    "set_matplotlib_style",
    "thesis_plotnine_theme",
]
