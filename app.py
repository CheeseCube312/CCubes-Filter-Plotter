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

# --- Constants ---
INTERP_GRID = np.arange(300, 1101, 5)  # 300–1100nm, 5nm step

# --- App Setup ---
st.set_page_config(page_title="CheeseCubes Filter Plotter", layout="wide")

# --- Sidebar ---
st.sidebar.header("Filter Plotter")
if st.sidebar.button("🔄 Refresh Filters"):
    st.cache_data.clear()
    st.rerun()

# --- Load Data from filters_data folder ---
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

            # Set hex color
            tmp["Hex"] = tmp["Hex Color"] if "Hex Color" in tmp.columns else "#1f77b4"

            # Set manufacturer
            tmp["Manufacturer"] = tmp["Manufacturer"] if "Manufacturer" in tmp.columns else "Unknown"

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
    return df, sorted(all_wls)

df, wavelengths = load_data()
if df.empty:
    st.stop()

filter_display = df["Filter Name"] + " (" + df["Filter Number"] + ", " + df["Manufacturer"] + ")"
display_to_index = {name: i for i, name in enumerate(filter_display)}

# --- Interpolation ---
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

# --- Sensor QE weighting curve (normalized for full-spectrum Bayer sensor) ---
qe_nm = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                  800, 850, 900, 950, 1000, 1050, 1100])
qe_val = np.array([0.00, 0.01, 0.10, 0.30, 0.65, 1.00, 0.95, 0.85, 0.70, 0.50,
                   0.35, 0.25, 0.15, 0.08, 0.03, 0.01, 0.00])
from scipy.interpolate import interp1d
sensor_qe = interp1d(qe_nm, qe_val, bounds_error=False, fill_value=0.0)(INTERP_GRID)


# --- Plotting UI Sidebar ---
selected = st.sidebar.multiselect("Select filters to plot", options=filter_display)
show_combined = st.sidebar.checkbox("Show combined filter", value=True)
log_stops = st.sidebar.checkbox("Display in stop-view", value=False)


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

    # --- Combined Filter ---
    if show_combined and len(selected_indices) > 1:
        stack = np.array([filter_matrix[i] for i in selected_indices])
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        combined = np.clip(combined, 1e-6, 1.0)

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

    st.plotly_chart(fig, use_container_width=True)

    # --- Estimate light loss ---
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

else:
    st.write("No filter selected right now.")




