import numpy as np


#matplotlib curve for exports
def add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
    ax.plot(
        x[~mask], y[~mask],
        label=label,
        linestyle='-', linewidth=1.75, color=color
    )
    if np.any(mask):
        ax.plot(
            x[mask], y[mask],
            linestyle='--', linewidth=1.0, color=color         
)
