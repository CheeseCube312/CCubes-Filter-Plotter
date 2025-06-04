#CheeseCubes Filter Plotter
#Fairly simple filter plotter
#I don't know how to code so this was made with lots of AI help
#ChatGPT, then VS Code + Ollama + Continue.dev

import glob
import os
import re
from pathlib import Path
from io import BytesIO

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

# --- Configuration ---
INTERP_GRID = np.arange(300, 1101, 5)  # Wavelength range: 300–1100nm, 5nm step
st.set_page_config(page_title="CheeseCubes Filter Plotter", layout="wide")


#def LOAD filter data
@st.cache_data
def load_filter_data():
    """
    Load all .tsv files from 'filters_data' and return combined DataFrame and wavelength list.
    """
    folder = "filters_data"
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, "*.tsv"))

    dfs, all_wls = [], set()

    for path in files:
        try:
            df = pd.read_csv(path, sep="\t")
            df.columns = [str(c).strip() for c in df.columns]
            wl_cols = [col for col in df.columns if col.isdigit()]

            if "Filter Number" not in df.columns:
                continue

            if "Hex Color" not in df.columns:
                df["Hex Color"] = "#1f77b4"
            df["Hex"] = df["Hex Color"]
            df["Manufacturer"] = df.get("Manufacturer", "Unknown")
            df["source_file"] = os.path.basename(path)

            all_wls.update(map(int, wl_cols))
            dfs.append(df)

        except Exception as e:
            st.warning(f"⚠️ Failed to load filter file '{os.path.basename(path)}': {e}")

    if not dfs:
        return pd.DataFrame(), []

    combined = pd.concat(dfs, ignore_index=True).copy()
    combined["Filter Number"] = combined["Filter Number"].astype(str)
    combined["Filter Name"] = combined.get("Filter Name", combined["Filter Number"])
    combined["is_lee"] = combined["source_file"].str.contains("LeeFilters", case=False)

    return combined, sorted(all_wls)


df, wavelengths = load_filter_data()
if df.empty:
    st.stop()
    st.error("No filter data found. Please add .tsv files to the 'filters_data/' folder.")

filter_display = df["Filter Name"] + " (" + df["Filter Number"] + ", " + df["Manufacturer"] + ")"
display_to_index = {name: i for i, name in enumerate(filter_display)}


#def LOAD QE data
@st.cache_data
def load_qe_data():
    """
    Load sensor quantum efficiency (QE) curves from 'QE_data'.
    """
    def normalize_channel(name):
        return {"red": "R", "green": "G", "blue": "B"}.get(name.strip().lower(), name.strip())

    folder = "QE_data"
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, "*.tsv"))

    qe_dict = {}
    default_key = None

    for path in files:
        try:
            df = pd.read_csv(path, sep="\t")
            wavelengths = df.columns[3:].astype(float)

            for _, row in df.iterrows():
                channel = normalize_channel(row["Channel"])
                brand = row["Camera Brand"].strip()
                model = row["Camera Model"].strip()
                key = f"{brand} {model}"

                spectrum = row.iloc[3:].astype(float).values
                interp_curve = np.interp(INTERP_GRID, wavelengths, spectrum, left=0, right=0)

                qe_dict.setdefault(key, {})[channel] = interp_curve

                if os.path.basename(path) == "Default_QE.tsv":
                    default_key = key

        except Exception as e:
            st.warning(f"⚠️ Failed to load QE file '{os.path.basename(path)}': {e}")

    return sorted(qe_dict.keys()), qe_dict, default_key


#def LOAD Illuminants
@st.cache_data
def load_illuminants():
    """
    Load spectral power distribution data for light sources from 'Illuminants'.
    """
    folder = "Illuminants"
    os.makedirs(folder, exist_ok=True)

    illuminants = {}
    for path in glob.glob(os.path.join(folder, "*.tsv")):
        try:
            df = pd.read_csv(path, sep="\t")

            if df.shape[1] < 2:
                st.warning(f"⚠️ Skipping '{path}': expected at least two columns.")
                continue

            wl = df.iloc[:, 0].astype(float).values
            power = df.iloc[:, 1].astype(float).values
            interp_curve = np.interp(INTERP_GRID, wl, power, left=0, right=0)

            name = Path(path).stem
            illuminants[name] = interp_curve

        except Exception as e:
            st.warning(f"⚠️ Failed to load illuminant '{path}': {e}")

    return illuminants


# --- Pre-Processing ---
def interpolate_filter_curves(df: pd.DataFrame, wavelengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate transmission curves of filters to a uniform wavelength grid.
    For Lee filters, mark regions beyond 700nm as extrapolated for visual styling only.

    Returns:
        - matrix: NxM array of interpolated transmissions
        - extrapolated_masks: NxM boolean array indicating extrapolated regions
    """
    matrix = []
    extrapolated_masks = []

    for _, row in df.iterrows():
        x = [wl for wl in wavelengths if str(wl) in row and not np.isnan(row[str(wl)])]
        y = [row[str(wl)] / 100 for wl in x]

        if len(x) < 2:
            # Not enough data to interpolate
            matrix.append(np.full_like(INTERP_GRID, np.nan, dtype=np.float32))
            extrapolated_masks.append(np.full_like(INTERP_GRID, False, dtype=bool))
            continue

        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
        interpolated = f(INTERP_GRID)
        mask = np.zeros_like(INTERP_GRID, dtype=bool)

        if row.get("is_lee", False):
            # Mark region above 700nm for dashed line styling
            mask = INTERP_GRID > 700

        matrix.append(interpolated)
        extrapolated_masks.append(mask)

    return np.array(matrix), np.array(extrapolated_masks)

filter_matrix, extrapolated_masks = interpolate_filter_curves(df, wavelengths)


#def calculate combined transmission
def get_combined_transmission(indices, filter_matrix, combine=True):
    if combine and len(indices) > 1:
        stack = np.array([filter_matrix[i] for i in indices])
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        return combined
    return filter_matrix[indices[0]]

# Sanitize Export filename
def sanitize_filename(name: str) -> str:
    """
    Replace any characters that are illegal in file/folder names.
    """
    return re.sub(r'[<>:"/\\|?*]', "-", name).strip()

# Core computation
def compute_weighted_response(trans, illum, current_qe, nan_safe=True):
    responses = {}
    for ch, qe in current_qe.items():
        if nan_safe:
            responses[ch] = np.sum(np.nan_to_num(trans * (qe / 100) * illum))
        else:
            valid = ~np.isnan(trans) & ~np.isnan(qe) & ~np.isnan(illum)
            if not np.any(valid):
                responses[ch] = np.nan
                continue
            responses[ch] = np.sum(trans[valid] * (qe[valid] / 100) * illum[valid])
    return responses

# Normalize RGB dict so G = 1.0
def normalize_rgb_to_green(rgb: dict[str, float]) -> dict[str, float]:
    """
    Normalize RGB dict so G = 1.0. If G is too small, defaults to safe values.
    """
    g = rgb.get("G", 0)
    if g > 1e-6:
        return {k: v / g for k, v in rgb.items()}
    return {"R": 0.0, "G": 1.0, "B": 0.0}

# Sanitize output folder names
def sanitize_folder_name(name: str) -> str:
    # Replace any illegal filesystem chars with “-”
    return re.sub(r'[<>:"/\\|?*]', "-", name).strip()


# ----Inline Code starts here

# --- Sidebar Filter Plotter ---
st.sidebar.header("Filter Plotter")
selected = st.sidebar.multiselect("Select filters to plot", options=filter_display)

# --- Sidebar Filter Multipliers ---
filter_multipliers = {}
if selected:
    with st.sidebar.expander("📦 Set Filter Stack Counts"):
        for name in selected:
            filter_multipliers[name] = st.number_input(
                f"{name}",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key=f"mult_{name}"
            )
# --- Sidebar show combined filter + display in stop view + refresh filters
show_combined = st.sidebar.checkbox("Show combined filter", value=True)
log_stops = st.sidebar.checkbox("Display in stop-view", value=False)

if st.sidebar.button("🔄 Refresh Filters"):
    st.cache_data.clear()
    st.rerun()


# Load illuminants
illuminants = load_illuminants()
illum_names = list(illuminants.keys())
if illum_names:
    default_index = illum_names.index("D65") if "D65" in illum_names else 0
    selected_illum_name = st.sidebar.selectbox("Scene Illuminant", illum_names, index=default_index)
    selected_illum = illuminants[selected_illum_name]
else:
    st.sidebar.error("⚠️ No illuminants found. Please add .tsv files to the 'Illuminants/' folder.")
    selected_illum_name = None
    selected_illum = None

# --- Sensor QE Profile Selection ---
camera_keys, qe_data, default_key = load_qe_data()
default_index = camera_keys.index(default_key) + 1 if default_key in camera_keys else 0

selected_camera = st.sidebar.selectbox(
    "Sensor QE Profile", ["None"] + camera_keys, index=default_index
)
current_qe = qe_data.get(selected_camera) if selected_camera != "None" else None

# Dynamic QE curve
if current_qe:
    # Average all selected QE curves
    qe_curves = np.array([curve for _, curve in current_qe.items()])
    sensor_qe = np.nanmean(qe_curves, axis=0)
else:
    # This should rarely happen — maybe warn?
    st.warning("No QE curves selected. Using flat QE response.")
    sensor_qe = np.ones_like(INTERP_GRID)

# App title
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h4 style='margin: 0;'>🧀 CheeseCubes Filter Designer</h4>
    </div>
    """,
    unsafe_allow_html=True
)

#--- Filters and Plotting ---
if selected:
    # Expand indices based on multiplier count
    selected_indices = []
    for name in selected:
        idx = display_to_index[name]
        count = filter_multipliers.get(name, 1)
        selected_indices.extend([idx] * count)

    # --- Combined Filter ---
    is_combined = show_combined and len(selected_indices) > 1
    combined = None

    if is_combined:
        trans = get_combined_transmission(selected_indices, filter_matrix, combine=True)
        combined = np.clip(trans, 1e-6, 1.0)
    else:
        trans = filter_matrix[selected_indices[0]]
        combined = None

    # --- Estimate light loss ---
    if is_combined:
        trans = combined
        label = "Combined"
    else:
        trans = filter_matrix[selected_indices[0]]
        label = df.iloc[selected_indices[0]]["Filter Name"]

    valid = ~np.isnan(trans)
    if np.any(valid):
        # Fallback in case QE is invalid
        if not np.any(sensor_qe):
            st.warning("⚠️ Sensor QE curve appears empty or invalid. Using default flat QE.")
            sensor_qe = np.ones_like(INTERP_GRID)

        avg_trans = np.average(np.clip(trans[valid], 1e-6, 1.0), weights=sensor_qe[valid])
        effective_stops = -np.log2(avg_trans)
        st.markdown(
            f"📉 **Estimated light loss ({label}):** {effective_stops:.2f} stops  \n"
            f"(Avg transmission: {avg_trans * 100:.1f}%)"
        )
    else:
        st.warning(f"⚠️ Cannot compute average transmission for {label}: insufficient data.")

    # --- Create plot ---
    fig = go.Figure()

    for idx in selected_indices:
        row = df.iloc[idx]
        transmission = np.clip(filter_matrix[idx], 1e-6, 1.0)
        extrap_mask = extrapolated_masks[idx]
        y_values = np.log2(transmission) if log_stops else transmission * 100

        fig.add_trace(go.Scatter(
            x=INTERP_GRID[~extrap_mask],
            y=y_values[~extrap_mask],
            name=f"{row['Filter Name']} ({row['Filter Number']})",
            mode="lines",
            line=dict(dash="solid", color=row["Hex Color"])
        ))

        if np.any(extrap_mask):
            fig.add_trace(go.Scatter(
                x=INTERP_GRID[extrap_mask],
                y=y_values[extrap_mask],
                name=f"{row['Filter Name']} ({row['Filter Number']}) (Extrapolated)",
                mode="lines",
                line=dict(dash="dash", color=row["Hex Color"]),
                showlegend=False
            ))

    if is_combined and combined is not None:
        combined_y = np.log2(combined) if log_stops else combined * 100
        fig.add_trace(go.Scatter(
            x=INTERP_GRID,
            y=combined_y,
            name="Combined Filter",
            mode="lines",
            line=dict(color="black", width=2)
        ))

    fig.update_layout(
        title="Combined Filter Response",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Stops (log₂)" if log_stops else "Transmission (%)",
        xaxis_range=(min(INTERP_GRID), max(INTERP_GRID)),
        yaxis=dict(
            range=(-10, 0) if log_stops else (0, 100),
            tickvals=[0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10] if log_stops else None,
            ticktext=["0", "-1", "-2", "-3", "-4", "-5", "-6", "-7", "-8", "-9", "-10"] if log_stops else None
        ),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Show sensor-weighted response (QE × Transmission) ---
if current_qe:

    st.subheader("Sensor-Weighted Response (QE × Transmission)")

    fig_response = go.Figure()
    apply_white_balance = st.checkbox("Apply White Balance to Response", value=False)

    # Determine transmission using helper
    if selected:
        trans = get_combined_transmission(selected_indices, filter_matrix, combine=show_combined)
    else:
        trans = np.ones_like(INTERP_GRID)  # Identity multiplier when no filters selected


    # Default WB gains
    white_balance_gains = {"R": 1.0, "G": 1.0, "B": 1.0}
    if apply_white_balance:
        white_balance_gains = st.session_state.get("white_balance_gains", white_balance_gains)

    # Channel toggle layout
    channel_names = list(current_qe.keys())
    cols = st.columns(len(channel_names))
    visible_channels = {}
    for i, channel in enumerate(channel_names):
        with cols[i]:
            visible_channels[channel] = st.checkbox(f"{channel} channel", value=True, key=f"resp_{channel}")

    # Initialize RGB response arrays
    r_response = np.zeros_like(INTERP_GRID, dtype=float)
    g_response = np.zeros_like(INTERP_GRID, dtype=float)
    b_response = np.zeros_like(INTERP_GRID, dtype=float)

    # Track peak response value
    max_response_value = 0

    # Plot each selected channel and build RGB
    for channel, show in visible_channels.items():
        if not show:
            continue

        qe_curve = current_qe[channel]
        gain = white_balance_gains.get(channel, 1.0)
        weighted = np.nan_to_num(trans * (qe_curve / 100))
        y_values = (weighted / gain) * 100 if apply_white_balance and gain > 0 else weighted * 100

        # Update peak value
        max_response_value = max(max_response_value, np.max(y_values))

        # Plot channel curve
        channel_colors = {"R": "red", "G": "green", "B": "blue"}
        fig_response.add_trace(go.Scatter(
            x=INTERP_GRID,
            y=y_values,
            name=f"{channel} Response{' (WB)' if apply_white_balance else ''}",
            mode="lines",
            line=dict(width=2, color=channel_colors.get(channel, "gray"))
        ))

        # Accumulate for RGB
        if channel == "R":
            r_response = y_values
        elif channel == "G":
            g_response = y_values
        elif channel == "B":
            b_response = y_values

    # Normalize RGB to 0–1, then convert to hex
    rgb = np.stack([r_response, g_response, b_response], axis=1)
    if (max_val := np.max(rgb)) > 0:
        rgb = rgb / max_val

    # Add a small minimum to each channel to prevent 0s from creating overly saturated artifacts
    rgb = np.clip(rgb, 1/255, 1)  # Prevents any channel from becoming pure zero


    def rgb_to_hex(row):
        r, g, b = (row * 255).astype(int)
        return f"#{r:02x}{g:02x}{b:02x}"

    gradient_colors = [rgb_to_hex(row) for row in rgb]

    # Add spectrum line just above the tallest curve
    spectrum_y = max_response_value * 1.05
    for i in range(len(INTERP_GRID) - 1):
        fig_response.add_trace(go.Scatter(
            x=[INTERP_GRID[i], INTERP_GRID[i+1] + 1e-6],
            y=[spectrum_y, spectrum_y],
            mode="lines",
            line=dict(color=gradient_colors[i], width=15),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Final layout update with autoscaling
    fig_response.update_layout(
        title="Effective Sensor Response (Transmission × QE)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Response (%)",
        xaxis_range=[300, 1100],
        yaxis_range=[0, spectrum_y * 1.05],
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig_response, use_container_width=True)




if selected and current_qe and selected_illum is not None and trans is not None:
    # --- Compute Raw RGB Response (Illuminant × QE × Transmission) ---
    rgb_response = {}
    for channel, qe_curve in current_qe.items():
        valid = ~np.isnan(trans) & ~np.isnan(qe_curve) & ~np.isnan(selected_illum)
        if not np.any(valid):
            st.warning(f"⚠️ No valid data for channel {channel}.")
            rgb_response[channel] = np.nan
            continue

        weighted = trans[valid] * (qe_curve[valid] / 100) * selected_illum[valid]
        rgb_response[channel] = np.sum(weighted)

    # --- Normalize to Green (Camera-Style Auto White Balance) ---
    g = rgb_response.get("G", 1.0)
    if g > 1e-6:
        white_balance_gains = {
            "R": rgb_response.get("R", 0) / g,
            "G": 1.0,
            "B": rgb_response.get("B", 0) / g,
        }
    else:
        white_balance_gains = {"R": 1.0, "G": 1.0, "B": 1.0}
        st.warning("⚠️ Green channel too low — fallback white balance used.")

    # --- Save WB Gains for plotting use ---
    st.session_state["white_balance_gains"] = white_balance_gains

    # --- Show WB Gains ---
    st.markdown(f"""
    **RGB White Balance Multipliers** (Green = 1.000):  
    R: {white_balance_gains['R']:.3f}   G: 1.000   B: {white_balance_gains['B']:.3f}
    """)

else:
    st.info("ℹ️ Select filters.")

# Export Button
if st.sidebar.button("📥 Download Report (PNG)"):
    # 0) Guards
    if not selected:
        st.warning("⚠️ No filters selected—nothing to export.")
    elif not current_qe:
        st.warning("⚠️ No QE profile selected—cannot build sensor response plot.")
    else:
        # 1) Build combo name
        combo_parts = []
        for name in selected:
            idx = display_to_index[name]
            row = df.iloc[idx]
            combo_parts.append(f"{row['Manufacturer']} {row['Filter Number']}")
        combo_name = ", ".join(combo_parts)
        safe_name = sanitize_filename(combo_name) or "report"

        # 2) Ensure output folder
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3) Rebuild selected_indices exactly as UI does
        selected_indices = []
        for name in selected:
            idx = display_to_index[name]
            count = st.session_state.get(f"mult_{name}", 1)
            selected_indices.extend([idx] * count)

        # 4) Combined vs single
        is_combined = show_combined and len(selected_indices) > 1
        if is_combined:
            trans = get_combined_transmission(selected_indices, filter_matrix, combine=True)
            combined = np.clip(trans, 1e-6, 1.0)
            trans = combined
            label = "Combined"
        else:
            trans = filter_matrix[selected_indices[0]]
            combined = None
            label = df.iloc[selected_indices[0]]["Filter Name"]

        # 5) Compute avg_trans & effective_stops
        valid = ~np.isnan(trans)
        if np.any(valid):
            sensor_qe = np.nanmean(np.array([curve for curve in current_qe.values()]), axis=0)
            avg_trans = np.average(np.clip(trans[valid], 1e-6, 1.0), weights=sensor_qe[valid])
            effective_stops = -np.log2(avg_trans)
        else:
            avg_trans = np.nan
            effective_stops = np.nan

        # 6) Retrieve WB gains from session (default to 1.0 if missing)
        wb = st.session_state.get("white_balance_gains", {"R": 1.0, "G": 1.0, "B": 1.0})

        # ------------------------------------------------
        # 7) Build Matplotlib figure with 5 rows, including spectrum strip
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

        # Row 0: FILTER INFO LIST
        ax0 = fig.add_subplot(gs[0])
        ax0.axis("off")
        ax0.text(0.01, 0.95, "Filters Used:", fontsize=12, fontweight="bold", va="top")

        # Start the list a bit lower so it doesn’t overlap the heading
        y = 0.75
        idx_counts = Counter(selected_indices)
        for idx, cnt in idx_counts.items():
            row = df.iloc[idx]
            brand = row["Manufacturer"]
            fname = row.get("Filter Name", "")
            fnum = row["Filter Number"]
            hexc = row["Hex Color"]

            # Draw color swatch
            swatch = Rectangle((0.01, y - 0.05), 0.03, 0.1, transform=ax0.transAxes,
                               facecolor=hexc, edgecolor="black", lw=0.5)
            ax0.add_patch(swatch)

            ax0.text(
                0.05, y,
                f"{brand} – {fname} (#{fnum})  {hexc}  × {cnt}",
                fontsize=10, va="center"
            )
            y -= 0.15
            if y < 0:
                break

        # Row 1: ESTIMATED LIGHT LOSS
        ax1 = fig.add_subplot(gs[1])
        ax1.axis("off")
        ax1.text(0.01, 0.85, "Estimated Light Loss:", fontsize=12, fontweight="bold", va="top")
        ax1.text(
            0.01, 0.30,
            f"{label} → {effective_stops:.2f} stops  (Avg: {avg_trans * 100:.1f}%)",
            fontsize=12, va="center"
        )

        # Row 2: FILTER TRANSMISSION (%) Plot
        ax2 = fig.add_subplot(gs[2])
        for idx in selected_indices:
            row = df.iloc[idx]
            trans_curve = np.clip(filter_matrix[idx], 1e-6, 1.0) * 100
            mask = extrapolated_masks[idx]
            ax2.plot(
                INTERP_GRID[~mask], trans_curve[~mask],
                label=f"{row['Filter Name']} ({row['Filter Number']})",
                linestyle='-', linewidth=1.75, color=row["Hex Color"]
            )
            if np.any(mask):
                ax2.plot(
                    INTERP_GRID[mask], trans_curve[mask],
                    linestyle='--', linewidth=1.0, color=row["Hex Color"]
                )

        if is_combined and combined is not None:
            combined_curve = combined * 100
            ax2.plot(
                INTERP_GRID, combined_curve,
                label="Combined Filter", color="black", linewidth=2.5
            )

        ax2.set_title("Filter Transmission (%)")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Transmission (%)")
        ax2.set_xlim(INTERP_GRID.min(), INTERP_GRID.max())
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right")

        # Row 3: RGB WHITE-BALANCE MULTIPLIERS
        ax3 = fig.add_subplot(gs[3])
        ax3.axis("off")
        ax3.text(0.01, 0.85, "RGB White Balance Multipliers:", fontsize=12, fontweight="bold", va="top")
        ax3.text(0.01, 0.40, f"R: {wb['R']:.3f}   G: 1.000   B: {wb['B']:.3f}", fontsize=12, va="center")

        # Row 4: SENSOR-WEIGHTED RESPONSE (QE × Transmission) + Spectrum Strip
        ax4 = fig.add_subplot(gs[4])
        max_response = 0.0

        # Prepare arrays for RGB‐based spectrum (same length as INTERP_GRID)
        r_resp = np.zeros_like(INTERP_GRID, dtype=float)
        g_resp = np.zeros_like(INTERP_GRID, dtype=float)
        b_resp = np.zeros_like(INTERP_GRID, dtype=float)

        # 1) Plot each channel’s response and accumulate into r_resp, g_resp, b_resp
        for channel, qe_curve in current_qe.items():
            wb_gain = wb.get(channel, 1.0)
            # Compute “raw” weighted signal (without WB):
            weighted_raw = np.nan_to_num(trans * (qe_curve / 100))

            # Now apply white balance by dividing (not multiplying):
            if wb_gain > 0:
                y_vals = (weighted_raw / wb_gain) * 100
            else:
                y_vals = weighted_raw * 100

            max_response = max(max_response, np.nanmax(y_vals, initial=0.0))

            color_map = {"R": "red", "G": "green", "B": "blue"}
            ax4.plot(
                INTERP_GRID, y_vals,
                label=f"{channel} Channel", linewidth=2, color=color_map.get(channel, "gray")
            )

            # Accumulate for the spectrum strip:
            if channel == "R":
                r_resp = y_vals
            elif channel == "G":
                g_resp = y_vals
            elif channel == "B":
                b_resp = y_vals

        # 2) Build normalized RGB matrix (N×3) for coloring the strip
        rgb_matrix = np.stack([r_resp, g_resp, b_resp], axis=1)  # shape (N, 3)
        if (mv := np.nanmax(rgb_matrix)) > 0:
            rgb_matrix = rgb_matrix / mv
        rgb_matrix = np.clip(rgb_matrix, 1/255, 1.0)

        # 3) Draw spectrum strip just above the curves using imshow
        strip_height = max_response * 0.05 if max_response > 0 else 1.0
        spectrum_bottom = max_response * 1.02
        spectrum_top = spectrum_bottom + strip_height
        ax4.imshow(
            rgb_matrix[np.newaxis, :, :],              # shape (1, N, 3)
            aspect='auto',
            extent=[INTERP_GRID.min(), INTERP_GRID.max(), spectrum_bottom, spectrum_top]
        )

        # 4) Finalize ax4 axes limits, labels, and move legend downward
        ax4.set_title("Sensor-Weighted Response (QE × Transmission)")
        ax4.set_xlabel("Wavelength (nm)")
        ax4.set_ylabel("Response (%)")
        ax4.set_xlim(INTERP_GRID.min(), INTERP_GRID.max())
        ax4.set_ylim(0, spectrum_top * 1.02 if spectrum_top > 0 else 1.0)
        ax4.legend(loc="upper right", bbox_to_anchor=(0.98, 0.85))

        # Finalize layout
        fig.tight_layout()

        # 8) Save to disk and offer download
        report_filename = f"{safe_name}.png"
        report_path = output_dir / report_filename
        fig.savefig(report_path, format="png", bbox_inches="tight")

        with open(report_path, "rb") as f:
            png_bytes = f.read()

        st.success(f"✔️ Report saved to: output/{report_filename}")
        st.download_button(
            "📥 Download Report PNG",
            data=png_bytes,
            file_name=report_filename,
            mime="image/png"
        )

# --- QE Plotting ---
if current_qe:
    with st.expander("Sensors Quantum Efficiency (QE) Curves", expanded=False):
        fig_qe = go.Figure()

        for channel, qe_curve in current_qe.items():
            channel_colors = {"R": "red", "G": "green", "B": "blue"}
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
