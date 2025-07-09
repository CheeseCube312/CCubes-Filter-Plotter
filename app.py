#CheeseCubes Filter Plotter
#I don't know how to code so this was made with lots of AI help
#ChatGPT, then VS Code + Ollama + Continue.dev


"""
app.py

Main orchestrator — ties together data loading, processing, visualization,
and user interface into a cohesive Streamlit application for interactive filter analysis.

--- Major Imports & Responsibilities ---
- glob, os, pathlib, io, warnings, re, collections.Counter  
    Filesystem and utility operations  
- numpy, pandas, scipy.interpolate, scipy.spatial.distance  
    Numerical processing, interpolation, and metrics  
- matplotlib (pyplot, GridSpec, Rectangle), plotly (io, graph_objects)  
    Static (PNG) and interactive (Plotly) plotting  
- streamlit  
    UI framework, session state, file uploads, download buttons  
- utils.advanced_search  
    Advanced filter search UI and logic  
- utils.exports.generate_report_png  
    Full-report PNG generator with swatches, plots, and metadata  
- utils.data_loader (load_filter_data, load_qe_data, load_illuminants)  
    TSV data ingestion, interpolation, and caching  
- utils.filter_math (compute_active_transmission, compute_combined_transmission, compute_filter_transmission, compute_rgb_response_from_transmission_and_qe)  
    Core filter‐to‐sensor light‐transmission math  
- utils.metrics (compute_effective_stops, compute_white_balance_gains, calculate_transmission_deviation_metrics)  
    Exposure loss, white balance gains, error metrics vs. target  
- utils.plotting.plotly_utils (create_filter_response_plot, create_sensor_response_plot, add_filter_curve_to_plotly)  
    Interactive Plotly figures for transmission and sensor response  
- utils.plotting.mpl_utils.add_filter_curve_to_matplotlib  
    Matplotlib helper for static curve plotting  
- utils.ui_components (ui_sidebar_filter_selection, ui_sidebar_filter_multipliers, ui_sidebar_extras, display_raw_qe_and_illuminant)  
    Reusable Streamlit sidebar and expander components  
- utils.file_utils.sanitize_filename_component  
    Safe filename generation  
- utils.importers.import_data  
    Frontend for CSV import of filters, illuminants, and QE data  
- utils.constants.INTERP_GRID  
    Shared wavelength grid (300–1100 nm, 1 nm steps)

--- Core Helpers Defined Here ---
- compute_selected_filter_indices(selected, multipliers, display_to_index, session_state):
    Builds the final list of filter indices honoring per-filter multipliers.
- convert_rgb_to_hex_color(row):
    Converts a normalized RGB vector to a hex color string for spectrum rendering.

--- Streamlit Layout ---
1. **Sidebar**  
   - Filter selection (multiselect + stack counts)  
   - Extras: illuminant, QE profile, reference target  
   - Report generation & download  
   - Settings: log‐stop view, channel toggles, cache rebuild, data import UI  

2. **Main Panel**  
   - Application title and status  
   - Light‐loss summary and deviation metrics  
   - Interactive Plotly plots: filter transmission & sensor response  
   - Static white‐balance gains display  
   - Optional raw QE and illuminant curves expander  

"""


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

<<<<<<< Updated upstream
=======
#import /utils/advanced_search.py 
>>>>>>> Stashed changes
from utils import advanced_search

#import /utils/generate_report_png
from utils.exports import generate_report_png

# --- Configuration ---
from utils.constants import INTERP_GRID
st.set_page_config(page_title="CheeseCubes Filter Plotter", layout="wide")

def _is_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

#BLOCK: Loaders + Interpolation
@st.cache_data
def load_filter_data():
    folder = os.path.join("data", "filters_data")
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, "*.tsv"))

    meta_list, matrix, masks = [], [], []

    for path in files:
        try:
            df = pd.read_csv(path, sep="\t")
            df.columns = [str(c).strip() for c in df.columns]

            # Detect wavelength columns
            wl_cols = sorted([float(c) for c in df.columns if _is_float(c)])
            str_wl_cols = [str(int(w)) for w in wl_cols]
            if not wl_cols or 'Filter Number' not in df.columns:
                continue

            for _, row in df.iterrows():
                fn = str(row['Filter Number'])
                name = row.get('Filter Name', fn)
                manufacturer = row.get('Manufacturer', 'Unknown')
                hex_color = row.get('Hex Color', '#1f77b4')
                is_lee = 'LeeFilters' in os.path.basename(path)

                raw = np.array([row.get(w, np.nan) for w in str_wl_cols], dtype=float)
                if raw.max() > 1.5:
                    raw /= 100.0

                interp_vals = np.interp(INTERP_GRID, wl_cols, raw, left=np.nan, right=np.nan)
                extrap_mask = (INTERP_GRID > 700) if is_lee else np.zeros_like(INTERP_GRID, dtype=bool)

                meta_list.append({
                    'Filter Number': fn,
                    'Filter Name': name,
                    'Manufacturer': manufacturer,
                    'Hex Color': hex_color,
                    'is_lee': is_lee
                })
                matrix.append(interp_vals)
                masks.append(extrap_mask)

        except Exception as e:
            st.warning(f"⚠️ Failed to load filter file '{os.path.basename(path)}': {e}")

    if not matrix:
        return pd.DataFrame(), np.empty((0, len(INTERP_GRID))), np.empty((0, len(INTERP_GRID)), dtype=bool)

    return pd.DataFrame(meta_list), np.vstack(matrix), np.vstack(masks)


@st.cache_data
def load_qe_data():
    folder = os.path.join('data', 'QE_data')
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, '*.tsv'))

    qe_dict = {}
    default_key = None

    for path in files:
        try:
            df = pd.read_csv(path, sep='\t')
            df.columns = [str(c).strip() for c in df.columns]

            wl_cols = sorted([float(c) for c in df.columns if _is_float(c)])
            str_wl_cols = [str(int(w)) for w in wl_cols]

            for _, row in df.iterrows():
                brand = row['Camera Brand'].strip()
                model = row['Camera Model'].strip()
                channel = row['Channel'].strip().upper()[0]
                key = f"{brand} {model}"

                raw = row[str_wl_cols].astype(float).values
                interp = np.interp(INTERP_GRID, wl_cols, raw, left=np.nan, right=np.nan)

                qe_dict.setdefault(key, {})[channel] = interp

                if os.path.basename(path) == 'Default_QE.tsv' and default_key is None:
                    default_key = key

        except Exception as e:
            st.warning(f"⚠️ Failed to load QE file '{os.path.basename(path)}': {e}")

    return sorted(qe_dict.keys()), qe_dict, default_key


@st.cache_data
def load_illuminants():
    folder = os.path.join('data', 'illuminants')
    os.makedirs(folder, exist_ok=True)

    illum, meta = {}, {}

    for path in glob.glob(os.path.join(folder, '*.tsv')):
        try:
            df = pd.read_csv(path, sep='\t')
            if df.shape[1] < 2:
                raise ValueError("File must have at least two columns (wavelength and power)")

            wl = df.iloc[:, 0].astype(float).values
            power = df.iloc[:, 1].astype(float).values

            interp = np.interp(INTERP_GRID, wl, power, left=np.nan, right=np.nan)
            name = os.path.splitext(os.path.basename(path))[0]

            illum[name] = interp
            if 'Description' in df.columns:
                meta[name] = df['Description'].dropna().iloc[0]

        except Exception as e:
            st.warning(f"⚠️ Failed to load illuminant '{os.path.basename(path)}': {e}")

    return illum, meta


#BLOCK: Sidebar UI Components
def ui_sidebar_filter_selection():

    current_selection = st.session_state.get("selected_filters", [])
    all_options = sorted(set(filter_display) | set(current_selection))

    selected = st.sidebar.multiselect(
        "Select filters to plot",
        options=all_options,
        default=current_selection,
        key="selected_filters",
    )

    # Advanced search toggle as checkbox
    advanced = st.sidebar.checkbox("Show Advanced Search", value=st.session_state.get("advanced", False))
    st.session_state.advanced = advanced

    if advanced:
        adv_result = advanced_search.advanced_filter_search(df, filter_matrix)
        if adv_result:
            new_selection = list(set(selected + adv_result))
            if set(new_selection) != set(selected):
                st.session_state["selected_filters"] = new_selection
                st.experimental_rerun()

    return selected

def ui_sidebar_filter_multipliers(selected):
    filter_multipliers = {}
    if selected:
        with st.sidebar.expander("📦 Set Filter Stack Counts", expanded=False):
            for name in selected:
                filter_multipliers[name] = st.number_input(
                    f"{name}",
                    min_value=1,
                    max_value=5,
                    value=1,
                    step=1,
                    key=f"mult_{name}"
                )
    return filter_multipliers

def ui_sidebar_extras(
    illuminants, metadata,
    camera_keys, qe_data, default_key,
    filter_display, df, filter_matrix, display_to_index
):
    with st.sidebar.expander("Extras", expanded=False):

        # --- Illuminant Selector ---
        if illuminants:
            illum_names = list(illuminants.keys())
            default_idx = illum_names.index("AM1.5_Global_REL") if "AM1.5_Global_REL" in illum_names else 0
            selected_illum_name = st.selectbox("Scene Illuminant", illum_names, index=default_idx)
            selected_illum = illuminants[selected_illum_name]
        else:
            st.error("⚠️ No illuminants found.")
            selected_illum_name, selected_illum = None, None

        # --- QE Profile Selector ---
        default_idx = camera_keys.index(default_key) + 1 if default_key in camera_keys else 0
        selected_camera = st.selectbox("Sensor QE Profile", ["None"] + camera_keys, index=default_idx)
        current_qe = qe_data.get(selected_camera) if selected_camera != "None" else None

        # --- Target Profile Selector ---
        target_options = ["None"] + list(filter_display)
        default_target = "None"
        target_selection = st.selectbox(
            "Reference Target",
            options=target_options,
            index=target_options.index(default_target),
            key="target_profile_selection"
        )

        target_profile = None
        if target_selection != "None":
            target_index = display_to_index[target_selection]
            row = df.iloc[target_index]

            raw_values = filter_matrix[target_index]
            valid = ~np.isnan(raw_values)

            target_profile = {
                "name": row["Filter Name"],
                "color": row.get("Hex Color", "666666"),
                "values": raw_values,
                "valid": valid
            }

    return selected_illum_name, selected_illum, current_qe, selected_camera, target_profile


#BLOCK: Filter Processing and Computation
def compute_filter_transmission(selected_indices, filter_matrix, df):

    if len(selected_indices) > 1:
        trans = compute_combined_transmission(selected_indices, filter_matrix, combine=True)
        trans = np.clip(trans, 1e-6, 1.0)
        label = "Combined"
        combined = trans
    else:
        trans = filter_matrix[selected_indices[0]]
        label = df.iloc[selected_indices[0]]["Filter Name"]
        combined = None

    return trans, label, combined

#Computes average transmission weighted by sensor QE and effective light loss in stops.
def compute_effective_stops(trans, sensor_qe):
    valid = ~np.isnan(trans) & ~np.isnan(sensor_qe)
    if not np.any(valid):
        return np.nan, np.nan

    clipped_trans = np.clip(trans[valid], 1e-6, 1.0)
    clipped_qe = sensor_qe[valid]

    if np.all(clipped_qe == 0):
        return np.nan, np.nan  # No effective weighting possible

    avg_trans = np.average(clipped_trans, weights=clipped_qe)
    effective_stops = -np.log2(avg_trans)

    return avg_trans, effective_stops


def compute_white_balance_gains(
    trans_interp: np.ndarray,
    current_qe: dict[str, np.ndarray],
    illum_curve: np.ndarray
) -> dict[str, float]:
    """
    Compute white balance gains normalized to green channel from transmission, QE, and illuminant.
    Returns a dict with gains such that green gain is exactly 1.0.
    """
    rgb_resp = {}
    for ch in ['R', 'G', 'B']:
        qe_curve = current_qe.get(ch)
        if qe_curve is None:
            rgb_resp[ch] = np.nan
            continue
        valid = ~np.isnan(trans_interp) & ~np.isnan(qe_curve) & ~np.isnan(illum_curve)
        if not valid.any():
            rgb_resp[ch] = np.nan
            continue
        rgb_resp[ch] = np.nansum(trans_interp[valid] * (qe_curve[valid] / 100) * illum_curve[valid])

    g = rgb_resp.get('G', np.nan)
    if not np.isnan(g) and g > 1e-6:
        # Normalize so green gain = 1.0
        return {ch: rgb_resp[ch] / g for ch in ['R', 'G', 'B']}
    
    st.warning("⚠️ Green channel too low — default white balance.")
    return {'R': 1.0, 'G': 1.0, 'B': 1.0}

def compute_selected_filter_indices(selected, multipliers, display_to_index, session_state):
    """Build the final list of selected filter indices with multiplier counts."""
    selected_indices = []
    for name in selected:
        idx = display_to_index[name]
        count = session_state.get(f"{multipliers}{name}", 1)
        selected_indices.extend([idx] * count)
    return selected_indices


def compute_active_transmission(selected, selected_indices=None, filter_matrix=None):
    if selected and selected_indices and filter_matrix is not None:
        return 	compute_combined_transmission(selected_indices, filter_matrix, combine=True)
    return np.ones_like(INTERP_GRID)  # Identity transmission (no filter effect)


#def calculate combined transmission
def compute_combined_transmission(indices, filter_matrix, combine=True):
    if combine and len(indices) > 1:
        stack = np.array([filter_matrix[i] for i in indices])
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        return combined
    return filter_matrix[indices[0]]


#BLOCK --- Consolidated Plotting and Metrics Functions ---
def create_filter_response_plot(
    interp_grid: np.ndarray,
    df,
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    selected_indices: list[int],
    combined: np.ndarray | None,
    target_profile: dict | None,
    log_stops: bool = False
) -> go.Figure:
    """
    Create Plotly figure for filter transmission curves, including individual, combined, and target.
    """
    fig = go.Figure()

    # Individual filter traces
    for idx in selected_indices:
        row = df.iloc[idx]
        trans_curve = np.clip(filter_matrix[idx], 1e-6, 1.0)
        mask = masks[idx]
        y = np.log2(trans_curve) if log_stops else trans_curve * 100

        # main trace
        fig.add_trace(go.Scatter(
            x=interp_grid[~mask], y=y[~mask],
            name=f"{row['Filter Name']} ({row['Filter Number']})",
            mode="lines",
            line=dict(dash="solid", color=row.get('Hex Color', 'black'))
        ))
        # extrapolated
        if mask.any():
            fig.add_trace(go.Scatter(
                x=interp_grid[mask], y=y[mask],
                name=f"{row['Filter Name']} ({row['Filter Number']}) (Extrap)",
                mode="lines",
                line=dict(dash="dash", color=row.get('Hex Color', 'black')),
                showlegend=False
            ))

    # Combined trace
    if combined is not None:
        y_combined = np.log2(combined) if log_stops else combined * 100
        fig.add_trace(go.Scatter(
            x=interp_grid, y=y_combined,
            name="Combined Filter",
            mode="lines",
            line=dict(color="black", width=2)
        ))

    # Target trace
    if target_profile:
        valid = target_profile['valid']
        vals = target_profile['values']
        y_target = np.log2(vals) if log_stops else vals * 100
        fig.add_trace(go.Scatter(
            x=interp_grid[valid], y=y_target[valid],
            name=f"Target: {target_profile['name']}",
            mode="lines",
            line=dict(color="black", dash="dot", width=2)
        ))

    # Layout
    y_title = "Stops (log₂)" if log_stops else "Transmission (%)"
    fig.update_layout(
        title="Combined Filter Response",
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_title,
        xaxis_range=(interp_grid.min(), interp_grid.max()),
        yaxis=dict(
            range=(-10, 0) if log_stops else (0, 100),
            tickvals=[-i for i in range(11)] if log_stops else None,
            ticktext=[f"-{i}" if i > 0 else "0" for i in range(11)] if log_stops else None
        ),
        showlegend=True
    )
    return fig

def display_raw_qe_and_illuminant(
    interp_grid: np.ndarray,
    current_qe: dict[str, np.ndarray] | None,
    illum_curve: np.ndarray | None,
    illum_name: str | None,
    metadata: dict[str, str],
    add_trace_fn
) -> None:
    """
    Display raw QE curves and illuminant in Streamlit expander.
    add_trace_fn is a function to add traces: (fig, x, y, mask, label, color).
    """
    QE_COLORS = {'R': 'red', 'G': 'green', 'B': 'blue'}

    with st.expander("📉 Show Raw QE and Illuminant Curves"):
        if current_qe:
            fig_qe = go.Figure()
            for ch, q in current_qe.items():
                mask = np.zeros_like(q, dtype=bool)
                add_trace_fn(fig_qe, interp_grid, q, mask, f"{ch} QE", QE_COLORS.get(ch, 'gray'))
            fig_qe.update_layout(
                title="Sensor Quantum Efficiency (QE)",
                xaxis_title="Wavelength (nm)", yaxis_title="QE (%)",
                xaxis_range=[300,1100], yaxis_range=[0,100], height=400
            )
            st.plotly_chart(fig_qe, use_container_width=True)
        else:
            st.info("ℹ️ No QE profile loaded.")

        if illum_curve is not None:
            fig_illum = go.Figure()
            fig_illum.add_trace(go.Scatter(
                x=interp_grid, y=illum_curve,
                mode="lines", name=f"Illuminant: {illum_name}",
                line=dict(color="orange", width=2)
            ))
            fig_illum.update_layout(
                title=f"Illuminant SPD: {illum_name}",
                xaxis_title="Wavelength (nm)", yaxis_title="Relative Power (%)",
                xaxis_range=[300,1100], yaxis_range=[0,105], height=400
            )
            st.plotly_chart(fig_illum, use_container_width=True)
            st.markdown(f"**Description:** {metadata.get(illum_name, '_No description available_')}")
        else:
            st.info("ℹ️ No illuminant profile loaded.")

def calculate_transmission_deviation_metrics(
    trans_curve: np.ndarray,
    target_profile: dict | None,  # allow None explicitly
    log_stops: bool = False
) -> dict:
    """
    Compute deviation metrics (MAE, Bias, Max Dev, RMSE) between trans_curve and target_profile.
    Returns an empty dict if target_profile is None or no valid overlap.
    """
    if target_profile is None:
        # No target given, return empty dict (or you could return None if preferred)
        return {}

    valid_t = ~np.isnan(trans_curve)
    valid_p = target_profile.get('valid')
    if valid_p is None:
        # Defensive: if 'valid' key missing, treat as no valid points
        return {}

    overlap = valid_t & valid_p
    if not overlap.any():
        return {}

    if log_stops:
        dev = np.log2(trans_curve[overlap]) - np.log2(target_profile['values'][overlap] / 100)
        unit = 'stops'
    else:
        dev = trans_curve[overlap] * 100 - target_profile['values'][overlap]
        unit = '%'

    mae = np.mean(np.abs(dev))
    bias = np.mean(dev)
    maxd = np.max(np.abs(dev))
    rmse = np.sqrt(np.mean(dev**2))

    return {'MAE': mae, 'Bias': bias, 'MaxDev': maxd, 'RMSE': rmse, 'Unit': unit}


def create_sensor_response_plot(
    interp_grid,
    trans_interp,
    qe_interp,
    visible_channels,
    white_balance_gains,
    apply_white_balance,
    target_profile=None,
    rgb_to_hex_fn=None,
    compute_sensor_weighted_rgb_response_fn=None,
):
    if rgb_to_hex_fn is None or compute_sensor_weighted_rgb_response_fn is None:
        raise ValueError("Required helper functions not provided.")

    responses, rgb_matrix, max_response = compute_sensor_weighted_rgb_response_fn(
        trans_interp, qe_interp, white_balance_gains, visible_channels
    )

    target_responses = None
    target_rgb_matrix = None
    target_max_response = 0

    if target_profile is not None:
        target_interp = target_profile["values"].copy()
        if np.nanmax(target_interp) > 1.5:
            target_interp /= 100.0
        target_responses, target_rgb_matrix, target_max_response = compute_sensor_weighted_rgb_response_fn(
            target_interp, qe_interp, white_balance_gains, visible_channels
        )

    fig = go.Figure()
    channel_colors = {"R": "red", "G": "green", "B": "blue"}

    for channel, show in visible_channels.items():
        if show:
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=responses[channel],
                name=f"{channel} Response (Filter){' (WB)' if apply_white_balance else ''}",
                mode="lines",
                line=dict(width=2, color=channel_colors.get(channel, "gray"))
            ))

    if target_responses is not None and len(target_responses) > 0:
        for channel, show in visible_channels.items():
            if show:
                fig.add_trace(go.Scatter(
                    x=interp_grid,
                    y=target_responses[channel],
                    name=f"{channel} Response (Target)",
                    mode="lines",
                    line=dict(width=2, color=channel_colors.get(channel, "gray"), dash="dot")
                ))

    combined_max_response = max(max_response, target_max_response)
    gradient_colors = [rgb_to_hex_fn(row) for row in rgb_matrix]
    spectrum_y = combined_max_response * 1.10

    for i in range(len(interp_grid) - 1):
        fig.add_trace(go.Scatter(
            x=[interp_grid[i], interp_grid[i + 1] + 1e-6],
            y=[spectrum_y, spectrum_y],
            mode="lines",
            line=dict(color=gradient_colors[i], width=15),
            showlegend=False,
            hoverinfo="skip"
        ))

    if target_responses is not None and len(target_responses) > 0:
        target_gradient_colors = [rgb_to_hex_fn(row) for row in target_rgb_matrix]
        target_spectrum_y = combined_max_response * 1.04

        for i in range(len(interp_grid) - 1):
            fig.add_trace(go.Scatter(
                x=[interp_grid[i], interp_grid[i + 1] + 1e-6],
                y=[target_spectrum_y, target_spectrum_y],
                mode="lines",
                line=dict(color=target_gradient_colors[i], width=10),
                showlegend=False,
                hoverinfo="skip"
            ))

        fig.add_annotation(
            x=interp_grid[-1],
            y=target_spectrum_y,
            text="Target Spectrum",
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor="right",
            bgcolor="rgba(0,0,0,0.4)",
        )

    fig.add_annotation(
        x=interp_grid[800],
        y=spectrum_y,
        text="Filter Spectrum",
        showarrow=False,
        font=dict(size=10, color="white"),
        xanchor="right",
        bgcolor="rgba(0,0,0,0.4)",
    )

    fig.update_layout(
        title="Effective Sensor Response (Transmission × QE) with Target Comparison",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Response (%)",
        xaxis_range=[300, 1100],
        yaxis_range=[0, combined_max_response * 1.15],
        showlegend=True,
        height=450
    )

    return fig

def compute_rgb_response_from_transmission_and_qe(trans, current_qe, white_balance_gains, visible_channels):
    responses = {}
    rgb_stack = []

    # Check if trans is valid (non-empty, non-None, contains some finite values)
    if trans is None or len(trans) == 0 or not np.any(np.isfinite(trans)):
        # Return zeros if no valid transmission data
        zero_array = np.zeros_like(next(iter(current_qe.values()))) if current_qe else np.array([])
        for channel in ['R', 'G', 'B']:
            responses[channel] = zero_array
            rgb_stack.append(zero_array)
        return responses, np.stack(rgb_stack, axis=1) if rgb_stack else np.array([]), 0.0

    max_response = 0.0

    for channel in ['R', 'G', 'B']:
        qe_curve = current_qe.get(channel)
        if qe_curve is None or len(qe_curve) != len(trans):
            responses[channel] = np.zeros_like(trans)
            rgb_stack.append(responses[channel])
            continue

        gain = white_balance_gains.get(channel, 1.0)
        # Avoid division by zero in gain
        if gain == 0:
            gain = 1.0

        weighted = np.nan_to_num(trans * (qe_curve / 100)) / gain * 100
        max_response = max(max_response, np.nanmax(weighted))

        if visible_channels.get(channel, True):
            responses[channel] = weighted
        else:
            responses[channel] = np.zeros_like(weighted)

        rgb_stack.append(responses[channel])

    rgb_matrix = np.stack(rgb_stack, axis=1)
    max_val = np.nanmax(rgb_matrix)
    if max_val > 0:
        rgb_matrix = rgb_matrix / max_val
    rgb_matrix = np.clip(rgb_matrix, 1 / 255, 1.0)

    return responses, rgb_matrix, max_response


#Visualization / Graphs handled here
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

def add_filter_curve_to_plotly(fig, x, y, mask, label, color):
    fig.add_trace(go.Scatter(
        x=x[~mask],
        y=y[~mask],
        name=label,
        mode="lines",
        line=dict(width=2, color=color)
    ))
    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            name=label + " (extrapolated)",
            mode="lines",
            line=dict(width=1, dash="dash", color=color),
            showlegend=False  # Optional: prevent legend clutter
        ))


def display_sensor_qe_and_illuminant_plot(selected_camera, current_qe, selected_illum_name, selected_illum, metadata):
    with st.expander("Sensors QE (QE) Curves and Illuminant", expanded=False):
        if current_qe:
            fig_qe = go.Figure()
            channel_colors = {"R": "red", "G": "green", "B": "blue"}
            for channel, qe_curve in current_qe.items():
                fig_qe.add_trace(go.Scatter(
                    x=INTERP_GRID,
                    y=qe_curve,
                    name=f"{channel} Channel",
                    mode="lines",
                    line=dict(width=2, color=channel_colors.get(channel, None))
                ))
            fig_qe.update_layout(
                title=f"Quantum Efficiency: {selected_camera or 'Unknown Camera'}",
                xaxis_title="Wavelength (nm)",
                yaxis_title="QE (%)",
                xaxis_range=[300, 1100],
                yaxis_range=[0, 100],
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_qe, use_container_width=True)

        if selected_illum is not None:
            illum_curve = selected_illum
            fig_illum = go.Figure()
            fig_illum.add_trace(go.Scatter(
                x=INTERP_GRID,
                y=illum_curve,
                mode="lines",
                name=f"Illuminant: {selected_illum_name}",
                line=dict(color="orange", width=2)
            ))
            fig_illum.update_layout(
                title=f"Illuminant Spectral Power Distribution: {selected_illum_name}",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Relative Power (%)",
                xaxis_range=[300, 1100],
                yaxis_range=[0, 105],
                height=250,
                margin=dict(t=30, b=20)
            )
            st.plotly_chart(fig_illum, use_container_width=True)

            st.markdown(f"**Illuminant Description:** {metadata.get(selected_illum_name, '_No description available_')}")

#BLOCK Misc uitilities
def convert_rgb_to_hex_color(row):
    r, g, b = (row * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"

#sanitize names
def sanitize_filename_component(name: str, lowercase=False, max_len=None) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', "-", name).strip()
    if lowercase:
        clean = clean.lower()
    return clean


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

# --- Settings: Toggles + Refresh ---
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
        st.cache_data.clear()
        st.experimental_rerun()


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


# Define standard colors for QE channels
QE_COLORS = {"R": "red", "G": "green", "B": "blue"}
# — Optional Viewer: Raw QE and Illuminant Curves —
display_raw_qe_and_illuminant(
    interp_grid=INTERP_GRID,
    current_qe=current_qe,
    illum_curve=selected_illum,
    illum_name=selected_illum_name,
    metadata=metadata,
    add_trace_fn=add_filter_curve_to_plotly
)
