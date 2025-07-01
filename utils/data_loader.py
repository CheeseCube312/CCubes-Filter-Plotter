import os
import glob
import numpy as np
import pandas as pd
import json
from pathlib import Path

import streamlit as st
from .constants import INTERP_GRID  


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
    files = glob.glob(os.path.join(folder, "**", "*.tsv"), recursive=True)

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

