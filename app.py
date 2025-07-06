#CheeseCubes Filter Plotter
#Fairly simple filter plotter
#I don't know how to code so this was made with lots of AI help
#ChatGPT, then VS Code + Ollama + Continue.dev

import glob
import os
import re
from pathlib import Path
from io import BytesIO
import io

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import Rectangle
from collections import Counter
import warnings

#import /utils/advanced_search.py and exports.py
from utils import advanced_search
from utils.exports import generate_report_png

#All data loaders: import /utils/data_loader.py
from utils.data_loader import load_filter_data, load_qe_data, load_illuminants

#Filter maths: import /utils/filter_math.py
from utils.filter_math import (
    compute_active_transmission,
    compute_combined_transmission,
    compute_filter_transmission,
    compute_rgb_response_from_transmission_and_qe,
)

#Metrics: import /utils/metrics.py
from utils.metrics import (
    compute_effective_stops,
    compute_white_balance_gains,
    calculate_transmission_deviation_metrics,
)

#import /utils/plotting/plotly_utils.py and mpl_utils.py
from utils.plotting.plotly_utils import (
    create_filter_response_plot,
    create_sensor_response_plot,
    add_filter_curve_to_plotly
)
from utils.plotting.mpl_utils import add_filter_curve_to_matplotlib

#UI components: import /utils/ui_components.py
from utils.ui_components import (
    ui_sidebar_filter_selection,
    ui_sidebar_filter_multipliers,
    ui_sidebar_extras,
    display_raw_qe_and_illuminant,
)

from utils.file_utils import sanitize_filename_component

#WebPlotDigitizer data converters
from utils.importers import import_data


# --- Configuration ---
from utils.constants import INTERP_GRID
st.set_page_config(page_title="CheeseCubes Filter Plotter", layout="wide")
CACHE_DIR = Path("cache")

def compute_selected_filter_indices(selected, multipliers, display_to_index, session_state):
    """Build the final list of selected filter indices with multiplier counts."""
    selected_indices = []
    for name in selected:
        idx = display_to_index[name]
        count = session_state.get(f"{multipliers}{name}", 1)
        selected_indices.extend([idx] * count)
    return selected_indices

#BLOCK Misc uitilities
def convert_rgb_to_hex_color(row):
    r, g, b = (row * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


#initialization
#Load filter metadata + spectra
df, filter_matrix, extrapolated_masks = load_filter_data()
if df.empty:
    st.error("No filter data found. Please add .tsv files to data/filters_data")
    st.stop()

#Build a human-readable list (and mapping) for the sidebar
filter_display = [
    f"{row['Filter Name']} ({row['Filter Number']}, {row['Manufacturer']})"
    for _, row in df.iterrows()
]
display_to_index = {name: idx for idx, name in enumerate(filter_display)}


# ----Inline Code starts here
# --- Sidebar ---
st.sidebar.header("Filter Plotter")

# Load illuminant and QE data early
illuminants, metadata = load_illuminants()
camera_keys, qe_data, default_key = load_qe_data()

# --- Filter Selection and Multiplier (TOP) ---
selected = ui_sidebar_filter_selection()
filter_multipliers = ui_sidebar_filter_multipliers(selected)

# --- QE + Illuminant + Target Profile (Extras Combined) ---
selected_illum_name, selected_illum, current_qe, selected_camera, target_profile = ui_sidebar_extras(
    illuminants, metadata,
    camera_keys, qe_data, default_key,
    filter_display, df, filter_matrix, display_to_index
)

# --- QE × Transmission Sensor Weighting ---
if current_qe:
    qe_curves = np.array([curve for _, curve in current_qe.items()])
    # Defensive check: non-empty and has any non-NaN values
    if qe_curves.size == 0 or not np.any(~np.isnan(qe_curves)):
        sensor_qe = np.ones_like(INTERP_GRID)  # fallback to all ones
    else:
        sensor_qe = np.nanmean(qe_curves, axis=0)
else:
    sensor_qe = np.ones_like(INTERP_GRID)

illum_curve = st.session_state.get("illum_curve_data")


# — Report Generation —
if st.sidebar.button("📄 Generate Report (PNG)"):
    # Compute selected indices once here
    selected_indices = compute_selected_filter_indices(
        selected, "mult_", display_to_index, st.session_state
    )
    if not selected_indices:
        st.warning("No filters selected – cannot generate report.")
    else:
        generate_report_png(
            selected_filters=selected,
            current_qe=current_qe,
            filter_matrix=filter_matrix,
            df=df,
            display_to_index=display_to_index,
            compute_selected_indices_fn=lambda sel: selected_indices,
            compute_filter_transmission_fn=lambda idxs: compute_filter_transmission(
                idxs, filter_matrix, df
            ),
            compute_effective_stops_fn=compute_effective_stops,
            compute_white_balance_gains_fn=compute_white_balance_gains,
            masks=extrapolated_masks,
            add_curve_fn=add_filter_curve_to_matplotlib,
            interp_grid=INTERP_GRID,
            sensor_qe=sensor_qe,
            camera_name=selected_camera or "UnknownCamera",
            illuminant_name=selected_illum_name or "UnknownIlluminant",
            sanitize_fn=sanitize_filename_component,
            illuminant_curve=illum_curve if illum_curve is not None else np.ones_like(INTERP_GRID),
        )

# Show download button if a report is ready
last_export = st.session_state.get("last_export", {})
if last_export.get("bytes"):
    st.sidebar.download_button(
        label="⬇️ Download Last Report",
        data=last_export["bytes"],
        file_name=last_export["name"],
        mime="image/png",
        use_container_width=True
    )

with st.sidebar.expander("Settings", expanded=False):
    # Log view toggle
    log_stops = st.checkbox("Display Filters in Stop View", value=False)

    # RGB channel toggles
    st.markdown("**Sensor-Weighted Response Channels**")
    rgb_channels = {}
    for channel in ["R", "G", "B"]:
        rgb_channels[channel] = st.checkbox(f"{channel} Channel", value=True, key=f"show_{channel}")

    # --- Rebuild Cache ---
    if st.button("🔄 Rebuild Filter Cache"):
        # Clear disk cache files
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*"):
                try:
                    f.unlink()
                except Exception as e:
                    st.warning(f"Failed to delete cache file {f}: {e}")

        # Clear Streamlit cached functions
        load_filter_data.clear()
        load_qe_data.clear()
        load_illuminants.clear()

        # Rerun the app so caches are rebuilt
        st.experimental_rerun()

    # --- Import Data Button ---
    if st.button("WebPlotDigitizer .csv importers"):
        st.session_state.show_import_data = True

    if st.session_state.get("show_import_data", False):
        from utils.importers.frontend_interface_importer import import_data
        st.markdown("---")
        import_data()


selected_indices = compute_selected_filter_indices(selected, "mult_", display_to_index, st.session_state) if selected else []

is_combined = len(selected_indices) > 1 if selected_indices else False

trans, label, combined = None, None, None
if selected_indices:
    trans, label, combined = compute_filter_transmission(selected_indices, filter_matrix, df)

# --- Title ---
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center;'>
    <h4 style='margin: 0;'>🧀 CheeseCubes Filter Designer</h4>
</div>
""", unsafe_allow_html=True)


# --- Filter Plotter (refactored) ---
if selected_indices:
    # Calculate combined transmission & label
    is_combined = len(selected_indices) > 1
    trans, label, combined = compute_filter_transmission(
        selected_indices, filter_matrix, df
    )

    # Light-loss summary
    valid = ~np.isnan(trans)
    if valid.any():
        avg_trans, effective_stops = compute_effective_stops(trans, sensor_qe)
        st.markdown(
            f"📉 **Estimated light loss ({label}):** "
            f"{effective_stops:.2f} stops  \n"
            f"(Avg transmission: {avg_trans * 100:.1f}%)"
        )
    else:
        st.warning(f"⚠️ Cannot compute average transmission for {label}: insufficient data.")

    # Plot individual, combined & target curves
    fig = create_filter_response_plot(
        interp_grid=INTERP_GRID,
        df=df,
        filter_matrix=filter_matrix,
        masks=extrapolated_masks,
        selected_indices=selected_indices,
        combined=combined,
        target_profile=target_profile,
        log_stops=log_stops,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Deviation from target
    trans_for_dev = combined if is_combined and combined is not None else trans
    metrics = calculate_transmission_deviation_metrics(trans_for_dev, target_profile, log_stops)
    if metrics:
        st.markdown(
            f"📊 **Deviation from target ({target_profile['name']}):**  \n"
            f"- MAE: `{metrics['MAE']:.2f} {metrics['Unit']}`  \n"
            f"- Bias: `{metrics['Bias']:.2f} {metrics['Unit']}`  \n"
            f"- Max Dev: `{metrics['MaxDev']:.2f} {metrics['Unit']}`  \n"
            f"- RMSE: `{metrics['RMSE']:.2f} {metrics['Unit']}`"
        )
    else:
        st.info("ℹ️ No valid overlap with target for deviation calculation.")


# Sensor-weighted QE
# --- Main Plot Block ---
if current_qe:
    st.subheader("Sensor-Weighted Response (QE × Transmission)")

    apply_white_balance = st.checkbox("Apply White Balance to Response", value=False)
    white_balance_gains = {"R": 1.0, "G": 1.0, "B": 1.0}
    if apply_white_balance:
        white_balance_gains = st.session_state.get("white_balance_gains", white_balance_gains)

    trans_interp = compute_active_transmission(selected, selected_indices, filter_matrix)

    fig_response = create_sensor_response_plot(
        interp_grid=INTERP_GRID,
        trans_interp=trans_interp,
        qe_interp=current_qe,
        visible_channels=rgb_channels,
        white_balance_gains=white_balance_gains,
        apply_white_balance=apply_white_balance,
        target_profile=target_profile,
        rgb_to_hex_fn=convert_rgb_to_hex_color,
        compute_sensor_weighted_rgb_response_fn=compute_rgb_response_from_transmission_and_qe
    )

    st.plotly_chart(fig_response, use_container_width=True)


# Make sure trans_interp is defined or fallback to ones if no filters selected
if selected and selected_indices:
    trans_interp = compute_active_transmission(selected, selected_indices, filter_matrix)
else:
    # No filters selected, use all ones vector matching INTERP_GRID length
    trans_interp = np.ones_like(INTERP_GRID)

if current_qe and selected_illum is not None:
    wb_gains = compute_white_balance_gains(
        trans_interp=trans_interp,
        current_qe=current_qe,
        illum_curve=selected_illum
    )
else:
    wb_gains = {'R': 1.0, 'G': 1.0, 'B': 1.0}

# — Compute & Display RGB White Balance —
if selected and current_qe and selected_illum is not None:
    wb_gains = compute_white_balance_gains(
        trans_interp=trans_interp,
        current_qe=current_qe,
        illum_curve=selected_illum
    )
    # store for reuse (e.g. in sensor‐response plotting)
    st.session_state["white_balance_gains"] = wb_gains

    st.markdown(
        f"**RGB Pre-White-Balance | relative channel intensity:** (Green = 1.000):  \n"
        f"R: {wb_gains['R']:.3f}   "
        f"G: {wb_gains['G']:.3f}   "
        f"B: {wb_gains['B']:.3f}"
    )
else:
    st.info("ℹ️ Select filters and a QE & illuminant profile to compute white balance.")


# — Optional Viewer: Raw QE and Illuminant Curves —
display_raw_qe_and_illuminant(
    interp_grid=INTERP_GRID,
    current_qe=current_qe,
    illum_curve=selected_illum,
    illum_name=selected_illum_name,
    metadata=metadata,
    add_trace_fn=add_filter_curve_to_plotly
)
