#CheeseCubes Filter Plotter
#Fairly simple filter plotter
#I don't know how to code so this was made with lots of AI help
#ChatGPT, then VS Code + Ollama + Continue.dev

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from pathlib import Path
from scipy.spatial import distance

# --- Constants ---
INTERP_GRID = np.arange(300, 1101, 5)  # 300–1100nm, 5nm step

# --- App Setup ---
st.set_page_config(page_title="CheeseCubes Filter Plotter", layout="wide")

# --- Create Sidebar ---
st.sidebar.header("Filter Plotter")
if st.sidebar.button("🔄 Refresh Filters"):
    st.cache_data.clear()
    st.rerun()

# --- DEF: Load Data from filters_data folder ---
@st.cache_data
def load_data():
    folder = "filters_data"
    os.makedirs(folder, exist_ok=True)
    dfs = []
    all_wls = set()
    files = glob.glob(os.path.join(folder, "*.tsv"))

    for path in files:
        try:
            tmp = pd.read_csv(path, sep="\t", header=0)
            tmp.columns = [str(c).strip() for c in tmp.columns]
            wl_cols = [c for c in tmp.columns if c.isdigit()]
            
            if "Filter Number" not in tmp.columns:
                continue

            if "Hex Color" not in tmp.columns:
                tmp["Hex Color"] = "#1f77b4"
            tmp["Hex"] = tmp["Hex Color"]

            if "Manufacturer" not in tmp.columns:
                tmp["Manufacturer"] = "Unknown"

            all_wls.update(int(w) for w in wl_cols)
            tmp["source_file"] = os.path.basename(path)
            dfs.append(tmp)
        except Exception as e:
            st.warning(f"Failed to load `{os.path.basename(path)}`: {e}")

    if not dfs:
        return pd.DataFrame(), []
    df = pd.concat(dfs, ignore_index=True)
    df["Filter Number"] = df["Filter Number"].astype(str)
    df["Filter Name"] = df.get("Filter Name", df["Filter Number"])
    df["is_lee"] = df["source_file"].str.contains("LeeFilters", case=False)
    df = df.copy()  # Defragment the DataFrame to prevent the warning

    return df, sorted(all_wls)

df, wavelengths = load_data()
if df.empty:
    st.stop()

filter_display = df["Filter Name"] + " (" + df["Filter Number"] + ", " + df["Manufacturer"] + ")"
display_to_index = {name: i for i, name in enumerate(filter_display)}



# -- DEF: Load data from QE_data folder --
@st.cache_data
def load_qe_files():
    qe_folder = "QE_data"
    files = glob.glob(os.path.join(qe_folder, "*.tsv"))
    qe_dict = {}
    default_key = None

    for f in files:
        df = pd.read_csv(f, sep="\t")
        for _, row in df.iterrows():
            channel = row["Channel"].strip().capitalize()

            # Normalize channel names
            channel_mapping = {"Red": "R", "Green": "G", "Blue": "B"}
            channel = channel_mapping.get(channel, channel)

            brand = row["Camera Brand"].strip()
            model = row["Camera Model"].strip()
            key = f"{brand} {model}"

            # Wavelength data
            spectrum = row.iloc[3:].astype(float).values
            wavelengths = df.columns[3:].astype(float)
            interp_curve = np.interp(INTERP_GRID, wavelengths, spectrum, left=0, right=0)

            if key not in qe_dict:
                qe_dict[key] = {}
            qe_dict[key][channel] = interp_curve

            # Check if this is the default file
            if os.path.basename(f) == "Default_QE.tsv":
                default_key = key
                
    keys_sorted = sorted(qe_dict.keys())
    return keys_sorted, qe_dict, default_key

# --- DEF: Load Illuminants ---
@st.cache_data
def load_illuminants():
    illum_folder = "Illuminants"
    os.makedirs(illum_folder, exist_ok=True)
    files = glob.glob(os.path.join(illum_folder, "*.tsv"))
    illuminants = {}
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
            wl = df.iloc[:, 0].values.astype(float)
            power = df.iloc[:, 1].values.astype(float)
            interp = np.interp(INTERP_GRID, wl, power, left=0, right=0)
            name = os.path.splitext(os.path.basename(f))[0]
            illuminants[name] = interp
        except Exception as e:
            st.warning(f"Failed to load illuminant {f}: {e}")
    return illuminants



# Load QE files
camera_keys, qe_data, default_key = load_qe_files()

# Set default selection
default_index = camera_keys.index(default_key) + 1 if default_key in camera_keys else 0

# Sidebar selector
selected_camera = st.sidebar.selectbox("Sensor QE Profile", ["None"] + camera_keys, index=default_index)

# Extract selected QE curves
if selected_camera != "None":
    current_qe = qe_data[selected_camera]
else:
    current_qe = None



# --- DEF: Interpolation ---
def interpolate_filters(df, wavelengths):
    matrix = []
    extrapolated_masks = []

    for _, row in df.iterrows():
        x = [wl for wl in wavelengths if str(wl) in row and not np.isnan(row[str(wl)])]
        y = [row[str(wl)] / 100 for wl in x]

        if len(x) < 2:
            matrix.append(np.full_like(INTERP_GRID, np.nan, dtype=np.float32))
            extrapolated_masks.append(np.full_like(INTERP_GRID, False, dtype=bool))
            continue

        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated = f(INTERP_GRID)

        extrapolated_mask = np.zeros_like(INTERP_GRID, dtype=bool)

        if row["is_lee"]:
            last_measured = max(x)
            to_800 = (INTERP_GRID > last_measured) & (INTERP_GRID <= 800)
            beyond_800 = INTERP_GRID > 800
            below_400 = INTERP_GRID < 400

            start_val = interpolated[INTERP_GRID == last_measured][0]
            interp_to_800 = np.linspace(start_val, 0.9, to_800.sum())
            interpolated[to_800] = interp_to_800
            interpolated[beyond_800] = 0.9
            interpolated[below_400] = np.nan

            extrapolated_mask[to_800 | beyond_800 | below_400] = True

        matrix.append(interpolated)
        extrapolated_masks.append(extrapolated_mask)

    return np.array(matrix), np.array(extrapolated_masks)

filter_matrix, extrapolated_masks = interpolate_filters(df, wavelengths)

# --- Dynamic QE weighting: fallback to default mono QE if no camera selected ---
if current_qe:
    # Use average of all selected QE channels
    qe_curves = np.array([curve for _, curve in current_qe.items()])
    sensor_qe = np.nanmean(qe_curves, axis=0)
else:
    # Default mono curve
    qe_nm = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                      800, 850, 900, 950, 1000, 1050, 1100])
    qe_val = np.array([0.00, 0.01, 0.10, 0.30, 0.65, 1.00, 0.95, 0.85, 0.70, 0.50,
                       0.35, 0.25, 0.15, 0.08, 0.03, 0.01, 0.00])
    sensor_qe = interp1d(qe_nm, qe_val, bounds_error=False, fill_value=0.0)(INTERP_GRID)


# --- Plotting UI Sidebar ---
selected = st.sidebar.multiselect("Select filters to plot", options=filter_display)
show_combined = st.sidebar.checkbox("Show combined filter", value=True)
log_stops = st.sidebar.checkbox("Display in stop-view", value=False)
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



# App title
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h4 style='margin: 0;'>🧀 CheeseCubes Filter Designer</h4>
    </div>
    """,
    unsafe_allow_html=True
)



if selected:
    selected_indices = [display_to_index[name] for name in selected]

    # --- Combined Filter ---
    combined = None  # Initialize outside to use later
    if show_combined and len(selected_indices) > 1:
        stack = np.array([filter_matrix[i] for i in selected_indices])
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        combined = np.clip(combined, 1e-6, 1.0)

    # --- Estimate light loss (show this up top) ---
    if show_combined and len(selected_indices) > 1:
        trans = combined
        label = "Combined"
    else:
        trans = filter_matrix[selected_indices[0]]
        label = df.iloc[selected_indices[0]]["Filter Name"]

    valid = ~np.isnan(trans)
    if np.any(valid):
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

    # --- Plot each selected filter ---
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

    # --- Plot combined line ---
    if combined is not None:
        combined_y = np.log2(combined) if log_stops else combined * 100
        fig.add_trace(go.Scatter(
            x=INTERP_GRID,
            y=combined_y,
            name="Combined Filter",
            mode="lines",
            line=dict(color="black", width=2)
        ))

    # --- Plot formatting ---
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

    # --- Show plot ---
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("No filter selected right now.")
# --- Show sensor-weighted response (QE × Transmission) ---
if selected and current_qe:
    st.subheader("Sensor-Weighted Response (QE × Transmission)")

    fig_response = go.Figure()
    trans = None

    if show_combined and len(selected_indices) > 1:
        # Use combined filter
        stack = np.array([filter_matrix[i] for i in selected_indices])
        trans = np.nanprod(stack, axis=0)
        trans[np.any(np.isnan(stack), axis=0)] = np.nan
    else:
        trans = filter_matrix[selected_indices[0]]

    # Horizontal layout for channel toggles
    channel_names = list(current_qe.keys())
    cols = st.columns(len(channel_names))

    for i, (channel, qe_curve) in enumerate(current_qe.items()):
        with cols[i]:
            show = st.checkbox(f"{channel} channel", value=True, key=f"resp_{channel}")
        if not show:
            continue

        weighted = trans * (qe_curve / 100)
        y_values = weighted * 100  # Always percentage view

        fig_response.add_trace(go.Scatter(
            x=INTERP_GRID,
            y=y_values,
            name=f"{channel} Response",
            mode="lines",
            line=dict(width=2)
        ))

    fig_response.update_layout(
        title="Effective Sensor Response (Transmission × QE)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Response (%)",
        xaxis_range=[300, 1100],
        yaxis_range=[0, 100],
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig_response, use_container_width=True)


# === White Balance + CCT Estimate ===
if selected and current_qe and selected_illum is not None:
    with st.expander("📏 Estimated White Balance (CCT & Tint)", expanded=True):
        st.markdown(f"Illuminant: **{selected_illum_name}**")

        # Use combined filter or single
        if show_combined and len(selected_indices) > 1:
            trans = np.nanprod(np.array([filter_matrix[i] for i in selected_indices]), axis=0)
        else:
            trans = filter_matrix[selected_indices[0]]

        # Multiply transmission × QE × illuminant per channel
        rgb = {}

        for channel, qe_curve in current_qe.items():
            weighted = np.nan_to_num(trans * (qe_curve / 100) * selected_illum)
            rgb[channel] = np.sum(weighted)

        
        # Normalize to G = 1 safely
        if "G" in rgb and rgb["G"] > 0:
            gains = {k: v / rgb["G"] for k, v in rgb.items()}
        else:
            st.warning("⚠️ Could not normalize RGB — missing or zero G channel.")
            gains = {"R": 0, "G": 1, "B": 0}


        # Convert to XYZ (sRGB primaries)
        M = np.array([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505]
        ])
        vec_rgb = np.array([gains.get("R", 0), 1.0, gains.get("B", 0)])
        XYZ = M @ vec_rgb
        X, Y, Z = XYZ
        total = X + Y + Z
        x = X / total
        y = Y / total

        # Estimate CCT using McCamy's approximation
        n = (x - 0.3320) / (y - 0.1858)
        CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

        # Compute tint as distance from Planckian locus (optional rough Δuv estimate)
        u = (4 * X) / (X + 15 * Y + 3 * Z)
        v = (6 * Y) / (X + 15 * Y + 3 * Z)
        duv = np.sqrt((u - 0.1978)**2 + (v - 0.4683)**2)  # approx reference locus near D65

        st.markdown(f"""
        **RGB White Balance Multipliers** (G = 1):  
        - R: `{gains.get('R', 0):.3f}`  
        - G: `1.000`  
        - B: `{gains.get('B', 0):.3f}`

        **Estimated Correlated Color Temperature (CCT)**: `{CCT:.0f}K`  
        **Tint (Δuv distance from locus)**: `{duv:.5f}`
        """)



 # --- QE Plotting ---
if current_qe:
    st.subheader("Quantum Efficiency (QE) Curves")

    fig_qe = go.Figure()

    for channel, qe_curve in current_qe.items():
        fig_qe.add_trace(go.Scatter(
            x=INTERP_GRID,
            y=qe_curve,
            name=f"{channel} Channel",
            mode="lines",
            line=dict(width=2)
        ))

    fig_qe.update_layout(
        title=f"Quantum Efficiency: {selected_camera}",
        xaxis_title="Wavelength (nm)",
        yaxis_title="QE (%)",
        xaxis_range=[300, 1100],
        yaxis_range=[0, 100],
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig_qe, use_container_width=True)
