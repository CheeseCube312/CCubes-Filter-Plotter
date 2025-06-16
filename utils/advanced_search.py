import numpy as np
import pandas as pd
import colorsys
import re
import streamlit as st
import plotly.graph_objects as go

# Constants from your constants module
from utils.constants import INTERP_GRID


# -- Filter & Sort Utilities -----------------------------------------

def filter_by_manufacturer(df, manufacturers):
    if not manufacturers:  # if empty list, return all
        return df
    return df[df["Manufacturer"].isin(manufacturers)]


def filter_by_trans_at_wavelength(
    df: pd.DataFrame,
    interp_grid: np.ndarray,
    matrix: np.ndarray,
    wavelength: int,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Filter filters by transmittance at a given wavelength (nm).
    Returns (filtered_df, trans_values_at_wl).
    """
    idx = np.where(interp_grid == wavelength)[0]
    if idx.size == 0:
        st.error(f"Wavelength {wavelength} nm not in INTERP_GRID")
        return df, np.zeros(len(df))
    i = idx[0]
    vals = matrix[:, i]
    mask = (vals >= min_t) & (vals <= max_t)
    return df[mask], vals[mask]


def color_swatch(hex_color: str, width: int = 22, height: int = 16) -> str:
    """
    Return HTML for a styled color swatch box.
    """
    return f"""
    <div style='
        display:inline-block;
        width:{width}px;
        height:{height}px;
        background-color:{hex_color};
        border:1px solid #aaa;
        border-radius:4px;
        margin:4px 0;
    '></div>
    """


def sort_by_hex_rainbow(df: pd.DataFrame, hex_col: str = "Hex Color") -> pd.DataFrame:
    """
    Sort DataFrame by hue, then saturation, then lightness extracted
    from its hex color column.
    """
    def hex_to_hsl(hex_str: str) -> tuple[float, float, float]:
        r, g, b = (int(hex_str[i:i+2], 16)/255.0 for i in (1, 3, 5))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h, s, l)

    # Identify and log invalid hex color rows
    valid_mask = df[hex_col].apply(is_valid_hex_color)
    invalid_rows = df[~valid_mask]

    if not invalid_rows.empty:
        st.warning(f"⚠ Found {len(invalid_rows)} filters with invalid hex color codes:")
        st.dataframe(invalid_rows[[hex_col, "Filter Number", "Filter Name", "Manufacturer"]])


    hsl_values = df[hex_col].apply(hex_to_hsl)
    hsl_df = pd.DataFrame(hsl_values.tolist(), columns=["_hue", "_sat", "_lit"], index=df.index)

    df_sorted = pd.concat([df, hsl_df], axis=1).sort_values(by=["_hue", "_sat", "_lit"])
    return df_sorted.drop(columns=["_hue", "_sat", "_lit"])



def sort_by_trans_at_wavelength(
    df: pd.DataFrame,
    trans_vals: np.ndarray,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Sort df by provided transmittance values array.
    """
    temp = df.copy()
    temp["_t"] = trans_vals
    sorted_df = temp.sort_values(by="_t", ascending=ascending).drop(columns=["_t"])
    return sorted_df


# -- Color helpers -----------------------------------------

def is_dark_color(hex_color: str) -> bool:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return luminance < 128  # Threshold for dark colors


def is_valid_hex_color(hex_code) -> bool:
    if not isinstance(hex_code, str):
        return False
    return bool(re.fullmatch(r"#([0-9a-fA-F]{6})", hex_code))


# -- Sparkline Generator ---------------------------------------------

def generate_sparkline_plotly(
    interp_grid: np.ndarray,
    full_matrix_row: np.ndarray,
    width: int = 750,  # narrower default width
    height: int = 250,
    show_axes: bool = True,
    line_color: str = "#1f77b4"
):
    y_scaled = full_matrix_row * 99 + 1  # maps 0 → 1 and 1 → 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=interp_grid,
        y=y_scaled,
        mode='lines',
        line=dict(color=line_color, width=2),
        hoverinfo='skip'
    ))

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=30 if show_axes else 0, r=10, t=10, b=30 if show_axes else 0),
        xaxis=dict(
            title='Wavelength (nm)' if show_axes else None,
            showgrid=show_axes,
            zeroline=False,
            showticklabels=show_axes,
            range=[interp_grid.min(), interp_grid.max()],
            showline=True,
            linewidth=1,
            linecolor='black',
        ),
        yaxis=dict(
            title='Transmittance (%)' if show_axes else None,
            showgrid=show_axes,
            zeroline=False,
            showticklabels=show_axes,
            range=[1, 100],
            showline=True,
            linewidth=1,
            linecolor='black',
        ),
        plot_bgcolor='white'
    )

    return fig


# -- Advanced Search UI ---------------------------------------------

def advanced_filter_search(
    df: pd.DataFrame,
    filter_matrix: np.ndarray,
):
    if not st.session_state.get("advanced", False):
        return

    with st.container():
        st.markdown("### 🔍 Advanced Filter Search")
        st.markdown("Use the controls below to search by manufacturer, color, or spectral transmittance.")

        cols = st.columns([2, 1, 2, 2, 1])
        manufs = cols[0].multiselect("Manufacturer", df["Manufacturer"].unique())
        wl = cols[1].number_input("λ (nm)", 300, 1100, 550, 5)
        tmin, tmax = cols[2].slider("Transmittance range (%)", 0, 100, (0, 100), step=1)
        sort_choice = cols[3].selectbox("Sort by", [
            "Filter Number", "Filter Name", "Hex‑Rainbow", f"Trans @ {wl} nm"
        ])
        apply_clicked = cols[4].button("🔄 Apply")

        if apply_clicked or "filters_applied" not in st.session_state:
            st.session_state["filters_applied"] = True
            st.session_state["manufs"] = manufs
            st.session_state["wl"] = wl
            st.session_state["tmin"] = tmin
            st.session_state["tmax"] = tmax
            st.session_state["sort_choice"] = sort_choice

        manufs = st.session_state.get("manufs", [])
        wl = st.session_state.get("wl", 550)
        tmin = st.session_state.get("tmin", 0)
        tmax = st.session_state.get("tmax", 100)
        sort_choice = st.session_state.get("sort_choice", "Filter Number")

        df_man = filter_by_manufacturer(df, manufs)
        df_filt, trans_vals = filter_by_trans_at_wavelength(
            df_man, INTERP_GRID, filter_matrix, wl, tmin / 100, tmax / 100
        )

        if sort_choice == "Hex‑Rainbow":
            df_sorted = sort_by_hex_rainbow(df_filt)
        elif sort_choice.startswith("Trans @"):
            df_sorted = sort_by_trans_at_wavelength(df_filt, trans_vals)
        elif sort_choice == "Filter Name":
            df_sorted = df_filt.sort_values("Filter Name")
        else:
            df_sorted = df_filt.sort_values("Filter Number")

        st.markdown("---")
        st.write(f"**{len(df_sorted)} filters found:**")

        for idx, row in df_sorted.iterrows():
            hex_color = row["Hex Color"]
            if not is_valid_hex_color(hex_color):
                hex_color = "#888888"

            number = row["Filter Number"]
            name = row["Filter Name"]
            brand = row["Manufacturer"]

            text_color = "#FFF" if is_dark_color(hex_color) else "#000"

            with st.container():
                cols = st.columns([6, 1])
                with cols[0]:
                    st.markdown(f"""
                        <div style="
                            background-color: {hex_color};
                            color: {text_color};
                            padding: 8px 12px;
                            border-radius: 6px;
                            font-weight: 600;
                            font-size: 1rem;
                            margin-bottom: 0;
                        ">
                            #{number} — {name} — {brand} — <code>{hex_color}</code>
                        </div>
                    """, unsafe_allow_html=True)

                toggle_key = f"filter_toggle_{idx}"
                with cols[1]:
                    show_details = st.toggle("Details", key=toggle_key, label_visibility="collapsed")

                if show_details:
                    fig = generate_sparkline_plotly(INTERP_GRID, filter_matrix[idx, :], line_color=hex_color)
                    st.plotly_chart(fig, use_container_width=False)
                    st.checkbox("Select this filter", key=f"adv_sel_{idx}")

        st.markdown("---")
        col_done, col_cancel = st.columns([1, 1])

        with col_done:
            if st.button("✅ Done"):
                selected_idxs = [
                    idx for idx in df_sorted.index
                    if st.session_state.get(f"adv_sel_{idx}", False)
                ]
                st.session_state.selected = selected_idxs
                st.session_state.advanced = False
                st.rerun()

        with col_cancel:
            if st.button("✖ Cancel"):
                st.session_state.advanced = False
                st.rerun()
