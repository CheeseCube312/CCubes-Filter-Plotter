import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import streamlit as st


def setup_matplotlib_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
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


def generate_report_png(
    selected_filters: list[str],
    current_qe: dict[str, np.ndarray],
    filter_matrix: np.ndarray,
    df,
    display_to_index: dict[str, int],
    compute_selected_indices_fn,
    compute_filter_transmission_fn,
    compute_effective_stops_fn,
    compute_white_balance_gains_fn,
    masks: np.ndarray,
    add_curve_fn,
    interp_grid: np.ndarray,
    sensor_qe: np.ndarray,
    camera_name: str,
    illuminant_name: str,
    sanitize_fn,
    illuminant_curve: np.ndarray
):
    # Guard
    if not selected_filters:
        st.warning("⚠️ No filters selected—nothing to export.")
        return

    # Sort combo name
    combo = []
    for name in sorted(selected_filters):
        idx = display_to_index.get(name)
        row = df.iloc[idx]
        combo.append((row['Manufacturer'], row['Filter Number'], row))
    combo_name = ", ".join(f"{m} {n}" for m, n, _ in combo)

    # Resolve indices
    selected_indices = compute_selected_indices_fn(selected_filters)
    if not selected_indices:
        st.warning("⚠️ Invalid filter selection—cannot resolve indices.")
        return

    # Compute curves
    trans, label, combined = compute_filter_transmission_fn(selected_indices)
    active_trans = combined if combined is not None else trans
    avg_trans, stops = compute_effective_stops_fn(active_trans, sensor_qe)
    wb = compute_white_balance_gains_fn(active_trans, current_qe, illuminant_curve)

    # Style & figure
    setup_matplotlib_style()
    fig = plt.figure(figsize=(8, 14), dpi=150, constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1.2, 0.6, 3.2, 0.8, 3.2])

    # 1: Filter swatches
    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    y0 = 0.9
    counts = {f: selected_filters.count(f) for f in set(selected_filters)}
    for name, cnt in counts.items():
        idx = display_to_index[name]
        row = df.iloc[idx]
        hexc = row.get('Hex Color', '#000000')
        rect = Rectangle((0.0, y0-0.15), 0.03, 0.1, transform=ax0.transAxes,
                         facecolor=hexc, edgecolor='black', lw=0.5)
        ax0.add_patch(rect)
        ax0.text(0.03, y0-0.1, f"{row['Manufacturer']} – {row['Filter Name']} (#{row['Filter Number']}) ×{cnt}",
                 transform=ax0.transAxes, fontsize=10, va='center')
        y0 -= 0.15

    # 2: Light loss
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    ax1.text(0.01, 0.7, 'Estimated Light Loss:', fontsize=12, fontweight='bold')
    ax1.text(0.01, 0.3, f"{label} → {stops:.2f} stops (Avg: {avg_trans*100:.1f}%)", fontsize=12)

    # 3: Transmission plot
    ax2 = fig.add_subplot(gs[2])
    for idx in selected_indices:
        row = df.iloc[idx]
        y = np.clip(filter_matrix[idx], 1e-6, 1.0) * 100
        mask = masks[idx]
        add_curve_fn(ax2, interp_grid, y, mask,
                     f"{row['Filter Name']} ({row['Filter Number']})", row.get('Hex Color', '#000000'))
    if len(selected_indices) > 1:
        ax2.plot(interp_grid, active_trans * 100, color='black', lw=2.5, label='Combined Filter')
    ax2.set_title('Filter Transmission (%)')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Transmission (%)')
    ax2.set_xlim(interp_grid.min(), interp_grid.max())
    ax2.set_ylim(0, 100)

    # 4: WB multipliers (convert gains to intensities)
    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')
    ax3.text(0.01, 0.6, 'White Balance Gains (Green = 1):', fontsize=12, fontweight='bold')

    # Convert gains back to raw intensities (relative to green)
    intensities = {
        'R': 1.0 / wb['R'] if wb['R'] != 0 else 0.0,
        'G': 1.0,
        'B': 1.0 / wb['B'] if wb['B'] != 0 else 0.0
    }

    ax3.text(0.01, 0.4, f"R: {intensities['R']:.3f}   G: {intensities['G']:.3f}   B: {intensities['B']:.3f}", fontsize=12)

    # 5: Sensor-weighted response
    ax4 = fig.add_subplot(gs[4])
    maxresp = 0
    stack = {}
    color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
    # Plot in correct RGB order
    for ch in ['R', 'G', 'B']:
        qe = current_qe.get(ch)
        if qe is None:
            continue
        gains = wb.get(ch, 1.0)
        resp = np.nan_to_num(active_trans * (qe / 100)) * 100 / gains
        ax4.plot(
            interp_grid,
            resp,
            label=f"{ch} Channel",
            lw=2,
            color=color_map[ch]
        )
        maxresp = max(maxresp, np.nanmax(resp))
        stack[ch] = resp

    # Spectrum strip
    rgb_matrix = np.stack([
        stack.get('R', np.zeros_like(active_trans)),
        stack.get('G', np.zeros_like(active_trans)),
        stack.get('B', np.zeros_like(active_trans))
    ], axis=1)
    mv = np.nanmax(rgb_matrix)
    if mv > 0:
        rgb_matrix /= mv
    extent = [interp_grid.min(), interp_grid.max(), maxresp * 1.02, maxresp * 1.07]
    ax4.imshow(rgb_matrix[np.newaxis, :, :], aspect='auto', extent=extent)

    ax4.set_title('Sensor-Weighted Response (White-Balanced)', fontsize=14, fontweight='bold')
    subtitle = f"Quantum Efficiency: {camera_name or 'None'}   |   Illuminant: {illuminant_name or 'None'}"
    ax4.text(0.5, 0.98,subtitle,transform=ax4.transAxes,ha='center',va='bottom',fontsize=8)
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Response (%)')
    ax4.set_xlim(interp_grid.min(), interp_grid.max())
    ax4.set_ylim(0, extent[3] * 1.02)
    ax4.legend(loc='upper right',fontsize=8,bbox_to_anchor=(1.0, 0.95)
)


    # Finalize
    fig.suptitle(f"Filter Report", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Filename
    fname = sanitize_fn(f"{camera_name}_{illuminant_name}_{combo_name}") + '.png'

    # NEW: Save to /outputs/[QE]/[Illuminant] folder
    import os
    output_dir = os.path.join("output", sanitize_fn(camera_name), sanitize_fn(illuminant_name))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

    # Session state for download button
    st.session_state['last_export'] = {'bytes': buf.getvalue(), 'name': fname}
    st.success(f"✔️ Report generated: {fname}")
