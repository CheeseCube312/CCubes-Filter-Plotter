import streamlit as st
import numpy as np
from plotly import graph_objs as go

from utils.constants import INTERP_GRID
from utils.plotting.plotly_utils import add_filter_curve_to_plotly
from utils.plotting.plotly_utils import QE_COLORS  # if needed

from utils import advanced_search

from utils.data_loader import load_filter_data

# Load data inside this module:
df, filter_matrix, extrapolated_masks = load_filter_data()

if df.empty:
    import streamlit as st
    st.error("No filter data found. Please add .tsv files to data/filters_data")
    st.stop()

filter_display = [
    f"{row['Filter Name']} ({row['Filter Number']}, {row['Manufacturer']})"
    for _, row in df.iterrows()
]

display_to_index = {name: idx for idx, name in enumerate(filter_display)}


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


#Expanders + Visuals

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

