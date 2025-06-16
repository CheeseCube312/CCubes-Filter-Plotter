import numpy as np
import pandas as pd
import colorsys
import re
import streamlit as st
import plotly.graph_objects as go

from utils.constants import INTERP_GRID


# -- Filter & Sort Utilities -----------------------------------------

def filter_by_manufacturer(df, manufacturers):
    return df if not manufacturers else df[df["Manufacturer"].isin(manufacturers)]

def filter_by_trans_at_wavelength(
    df: pd.DataFrame,
    interp_grid: np.ndarray,
    matrix: np.ndarray,
    wavelength: int,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    idx = np.where(interp_grid == wavelength)[0]
    if idx.size == 0:
        st.error(f"Wavelength {wavelength} nm not in INTERP_GRID")
        return df, np.zeros(len(df))
    i = idx[0]

    # Make sure we're filtering the correct rows in the matrix
    df_indices = df.index.to_numpy()
    vals = matrix[df_indices, i]

    mask = (vals >= min_t) & (vals <= max_t)

    return df.iloc[mask], vals[mask]


def color_swatch(hex_color, width=22, height=16):
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

def sort_by_hex_rainbow(df, hex_col="Hex Color"):
    def hex_to_hsl(hex_str):
        r, g, b = (int(hex_str[i:i+2], 16)/255.0 for i in (1, 3, 5))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h, s, l)

    valid_mask = df[hex_col].apply(is_valid_hex_color)
    invalid_rows = df[~valid_mask]
    if not invalid_rows.empty:
        st.warning(f"⚠ Found {len(invalid_rows)} filters with invalid hex color codes:")
        st.dataframe(invalid_rows[[hex_col, "Filter Number", "Filter Name", "Manufacturer"]])

    hsl_df = df[hex_col].apply(hex_to_hsl).apply(pd.Series)
    hsl_df.columns = ["_hue", "_sat", "_lit"]
    df_sorted = pd.concat([df, hsl_df], axis=1).sort_values(by=["_hue", "_sat", "_lit"])
    return df_sorted.drop(columns=["_hue", "_sat", "_lit"])

def sort_by_trans_at_wavelength(df, trans_vals, ascending=False):
    temp = df.copy()
    temp["_t"] = trans_vals
    return temp.sort_values(by="_t", ascending=ascending).drop(columns=["_t"])


# -- Color Helpers -----------------------------------------

def is_dark_color(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return luminance < 128

def is_valid_hex_color(hex_code):
    return isinstance(hex_code, str) and bool(re.fullmatch(r"#([0-9a-fA-F]{6})", hex_code))


# -- Sparkline -----------------------------------------

def generate_sparkline_plotly(interp_grid, full_matrix_row, width=750, height=250, show_axes=True, line_color="#1f77b4"):
    y_scaled = full_matrix_row * 99 + 1
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
        xaxis=dict(title='Wavelength (nm)' if show_axes else None, showgrid=show_axes, zeroline=False, showticklabels=show_axes, range=[interp_grid.min(), interp_grid.max()], showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(title='Transmittance (%)' if show_axes else None, showgrid=show_axes, zeroline=False, showticklabels=show_axes, range=[1, 100], showline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='white'
    )
    return fig


# -- Advanced Search UI -----------------------------------------

def advanced_filter_search(df: pd.DataFrame, filter_matrix: np.ndarray):
    if not st.session_state.get("advanced", False):
        return

    st.markdown("### 🔍 Advanced Filter Search")
    st.markdown("Use the controls below to search by manufacturer, color, or spectral transmittance.")

    with st.form("adv_search_form"):
        cols = st.columns([2, 1, 2, 2, 1])
        manufs = cols[0].multiselect("Manufacturer", df["Manufacturer"].unique())
        wl = cols[1].number_input("λ (nm)", 300, 1100, 550, 5)
        tmin, tmax = cols[2].slider("Transmittance range (%)", 0, 100, (0, 100), step=1)
        sort_choice = cols[3].selectbox("Sort by", [
            "Filter Number", "Filter Name", "Hex‑Rainbow", f"Trans @ {wl} nm"
        ])
        apply_clicked = cols[4].form_submit_button("🔄 Apply")

    if apply_clicked:
        st.session_state.update({
            "filters_applied": True,
            "manufs": manufs,
            "wl": wl,
            "tmin": tmin,
            "tmax": tmax,
            "sort_choice": sort_choice
        })


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
                        {number} — {name} — {brand} 
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
            selected_display = [
                f"{df_sorted.loc[idx, 'Filter Name']} "
                f"({df_sorted.loc[idx, 'Filter Number']}, "
                f"{df_sorted.loc[idx, 'Manufacturer']})"
                for idx in selected_idxs
            ]

            current = st.session_state.get("selected_filters", [])
            st.session_state["selected_filters"] = list(set(current + selected_display))
            st.session_state.advanced = False
            st.rerun()

    with col_cancel:
        if st.button("✖ Cancel"):
            st.session_state.advanced = False
            st.rerun()
