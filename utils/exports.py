import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from collections import Counter
from pathlib import Path
import warnings
import streamlit as st

def generate_report_png(
    selected, current_qe, filter_matrix, df, display_to_index,
    get_selected_indices, compute_transmission, compute_average_stops,
    normalize_rgb_to_green, extrapolated_masks, add_filter_trace_matplotlib,
    INTERP_GRID, sensor_qe, selected_camera, selected_illum_name,
    sanitize_path_part
):
    # Guards
    if not selected:
        st.warning("⚠️ No filters selected—nothing to export.")
        return None
    elif not current_qe:
        st.warning("⚠️ No QE profile selected—cannot build sensor response plot.")
        return None

    # Build sorted, grouped combo name
    combo_rows = []
    for name in selected:
        idx = display_to_index[name]
        row = df.iloc[idx]
        combo_rows.append((row["Manufacturer"], row["Filter Number"], row))
    combo_rows.sort(key=lambda x: (x[0], x[1]))
    combo_name = ", ".join(f"{row['Manufacturer']} {row['Filter Number']}" for _, _, row in combo_rows)

    # Build selected_indices via helper
    selected_indices = get_selected_indices(selected, "mult_", display_to_index, st.session_state)

    # Compute transmission & label via helper
    trans, label, combined = compute_transmission(selected_indices, filter_matrix, df)

    # Compute avg_trans & effective stops via helper
    avg_trans, effective_stops = compute_average_stops(trans, sensor_qe)

    # Get normalized white balance gains
    raw_wb = st.session_state.get("white_balance_gains", {"R": 1.0, "G": 1.0, "B": 1.0})
    wb = normalize_rgb_to_green(raw_wb)

    # Setup matplotlib styling & figure
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#EEEEEE",
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })

    fig = plt.figure(figsize=(8, 14), dpi=150, constrained_layout=True)
    gs = GridSpec(5, 1, height_ratios=[1.2, 0.6, 3.2, 0.8, 3.2], figure=fig)

    # FILTER INFO LIST
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.text(0.01, 0.95, "Filters Used:", fontsize=12, fontweight="bold", va="top")
    y = 0.75
    idx_counts = Counter(selected_indices)
    for idx, cnt in idx_counts.items():
        row = df.iloc[idx]
        brand = row["Manufacturer"]
        fname = row.get("Filter Name", "")
        fnum = row["Filter Number"]
        hexc = row["Hex Color"]
        swatch = Rectangle((0.01, y - 0.05), 0.03, 0.1, transform=ax0.transAxes,
                           facecolor=hexc, edgecolor="black", lw=0.5)
        ax0.add_patch(swatch)
        ax0.text(0.05, y, f"{brand} – {fname} (#{fnum})  × {cnt}", fontsize=10, va="center")
        y -= 0.15
        if y < 0:
            break

    # ESTIMATED LIGHT LOSS
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.text(0.01, 0.85, "Estimated Light Loss:", fontsize=12, fontweight="bold", va="top")
    ax1.text(0.01, 0.30, f"{label} → {effective_stops:.2f} stops  (Avg: {avg_trans * 100:.1f}%)",
             fontsize=12, va="center")

    # FILTER TRANSMISSION Plot
    ax2 = fig.add_subplot(gs[2])
    for idx in selected_indices:
        row = df.iloc[idx]
        trans_curve = np.clip(filter_matrix[idx], 1e-6, 1.0) * 100
        mask = extrapolated_masks[idx]
        add_filter_trace_matplotlib(
            ax2, INTERP_GRID, trans_curve, mask,
            f"{row['Filter Name']} ({row['Filter Number']})",
            row["Hex Color"]
        )

    if len(selected_indices) > 1:
        combined_curve = trans * 100
        ax2.plot(INTERP_GRID, combined_curve, label="Combined Filter", color="black", linewidth=2.5)

    ax2.set_title("Filter Transmission (%)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmission (%)")
    ax2.set_xlim(INTERP_GRID.min(), INTERP_GRID.max())
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")

    # RGB WHITE-BALANCE MULTIPLIERS
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    ax3.text(0.01, 0.85, "RGB White Balance Multipliers:", fontsize=12, fontweight="bold", va="top")
    ax3.text(0.01, 0.40, f"R: {wb['R']:.3f}   G: 1.000   B: {wb['B']:.3f}", fontsize=12, va="center")

    # SENSOR-WEIGHTED RESPONSE + Spectrum Strip
    ax4 = fig.add_subplot(gs[4])
    max_response = 0.0

    r_resp = np.zeros_like(INTERP_GRID, dtype=float)
    g_resp = np.zeros_like(INTERP_GRID, dtype=float)
    b_resp = np.zeros_like(INTERP_GRID, dtype=float)

    for channel, qe_curve in current_qe.items():
        wb_gain = wb.get(channel, 1.0)
        weighted_raw = np.nan_to_num(trans * (qe_curve / 100))

        if wb_gain > 0:
            y_vals = weighted_raw * wb_gain * 100
        else:
            y_vals = weighted_raw * 100

        max_response = max(max_response, np.nanmax(y_vals, initial=0.0))
        color_map = {"R": "red", "G": "green", "B": "blue"}
        ax4.plot(INTERP_GRID, y_vals, label=f"{channel} Channel", linewidth=2, color=color_map.get(channel, "gray"))
        if channel == "R":
            r_resp = y_vals
        elif channel == "G":
            g_resp = y_vals
        elif channel == "B":
            b_resp = y_vals

    rgb_matrix = np.stack([r_resp, g_resp, b_resp], axis=1)
    mv = np.nanmax(rgb_matrix)
    if mv > 0:
        rgb_matrix /= mv
    rgb_matrix = np.clip(rgb_matrix, 1/255, 1.0)

    strip_height = max_response * 0.05 if max_response > 0 else 1.0
    spectrum_bottom = max_response * 1.02
    spectrum_top = spectrum_bottom + strip_height
    ax4.imshow(rgb_matrix[np.newaxis, :, :], aspect='auto',
               extent=[INTERP_GRID.min(), INTERP_GRID.max(), spectrum_bottom, spectrum_top])

    ax4.set_title("Sensor-Weighted Response (White-Balanced)", fontsize=14, fontweight="bold", loc="center", pad=10)

    qe_label = f"Quantum Efficiency data: {selected_camera or 'None'}"
    illum_label = f"Illuminant: {selected_illum_name or 'None'}"
    subtitle = f"{qe_label}   |   {illum_label}"
    ax4.text(0.5, 0.99, subtitle, transform=ax4.transAxes, fontsize=8,
             fontweight="normal", ha="center", va="bottom")

    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_ylabel("Response (%)")
    ax4.set_xlim(INTERP_GRID.min(), INTERP_GRID.max())
    ax4.set_ylim(0, spectrum_top * 1.02 if spectrum_top > 0 else 1.0)
    ax4.legend(loc="upper right", bbox_to_anchor=(0.98, 0.85))

    fig.tight_layout()

    # Unified output directory setup
    base_output = Path("output")
    qe_safe = sanitize_path_part(selected_camera or "Unknown_QE")
    illum_safe = sanitize_path_part(selected_illum_name or "Unknown_Illuminant")
    filters_safe = sanitize_path_part(combo_name, max_len=60)

    output_dir = base_output / f"QE_{qe_safe}" / f"Illuminant_{illum_safe}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_filename = f"{filters_safe}.png"
    report_path = output_dir / report_filename

    # Save figure, suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.savefig(report_path, format="png", bbox_inches="tight")

    with open(report_path, "rb") as f:
        png_bytes = f.read()

    st.success(f"✔️ Report saved to: {report_path}")
    st.session_state["last_export"] = {
        "path": report_path,
        "name": report_filename,
        "bytes": png_bytes
    }

    return report_path
