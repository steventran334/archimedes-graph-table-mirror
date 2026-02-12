import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
import io
import unicodedata
import re
import numpy as np

# Standard Plotting Setup
st.set_page_config(layout="wide", page_title="Archimedes Plotter")

st.title("Archimedes to CSV + Mirrored Buoyancy Plot")

st.markdown("""
    <style>
        .main .block-container {
            max-width: 100% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stButton button {
            width: 100%;
            padding: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---
def normalize(s):
    return unicodedata.normalize("NFKD", s.replace("μ", "u")).encode("ascii", "ignore").decode("utf-8").strip().lower()

def extract_value(lines, key, key_index=1, value_index=2):
    norm_key = normalize(key)
    for line in lines:
        parts = [x.strip() for x in line.split(',')]
        if len(parts) > key_index and normalize(parts[key_index]) == norm_key:
            if len(parts) > value_index and parts[value_index].strip():
                return parts[value_index]
            else:
                return "(empty)"
    return "N/A"

def find_index(lines, start_text):
    return next((i for i, line in enumerate(lines) if line.strip().startswith(start_text)), None)

def render_table_as_figure(df, col_width=3.0, row_height=0.625, font_size=14):
    fig, ax = plt.subplots(figsize=(col_width * (df.shape[1] + 1), row_height * (df.shape[0] + 1)))
    ax.axis('off')
    mpl_table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center', cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    mpl_table.scale(1.2, 1.2)
    return fig

# --- 1. File Upload ---
raw_uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if raw_uploaded_files:
    file_dict = {f.name: f for f in raw_uploaded_files}
    file_names = list(file_dict.keys())

    # --- 2. Native Reordering Logic ---
    st.subheader("Step 1: Define File Order & Selection")
    
    if 'order' not in st.session_state:
        st.session_state.order = []

    # UI for selection and movement
    col_sel, col_move = st.columns([1, 1])
    
    with col_sel:
        selected = st.multiselect("Pick files to include:", options=file_names, default=None)
        # Sync state: remove deselected, add new ones to bottom
        st.session_state.order = [f for f in st.session_state.order if f in selected]
        for f in selected:
            if f not in st.session_state.order:
                st.session_state.order.append(f)

    with col_move:
        st.write("Rearrange Order:")
        current_list = list(st.session_state.order)
        for i, name in enumerate(current_list):
            r_col1, r_col2, r_col3 = st.columns([4, 1, 1])
            r_col1.caption(f"**{name}**")
            if r_col2.button("↑", key=f"up_{name}") and i > 0:
                current_list[i], current_list[i-1] = current_list[i-1], current_list[i]
                st.session_state.order = current_list
                st.rerun()
            if r_col3.button("↓", key=f"down_{name}") and i < len(current_list)-1:
                current_list[i], current_list[i+1] = current_list[i+1], current_list[i]
                st.session_state.order = current_list
                st.rerun()

    # Final list based on user reordering
    final_files = [file_dict[name] for name in st.session_state.order]

    if final_files:
        all_summaries = {}
        histogram_data = []
        dataset_labels = {}

        st.subheader("Step 2: Customize Labels")
        for f in final_files:
            default_label = f.name.rsplit(".", 1)[0]
            dataset_labels[f.name] = st.text_input(f"Label for {f.name}", value=default_label, key=f"lbl_{f.name}")

        # --- 3. Processing ---
        for uploaded_file in final_files:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode("utf-8").splitlines()
            
            idx_summary = find_index(content, "SUMMARY DATA")
            idx_filters = find_index(content, "PARTICLE FILTERS")
            idx_stats = find_index(content, "SUMMARY STATISTICS")
            idx_dist_header = find_index(content, "Bin Start")

            if idx_dist_header is None: continue

            summary_data = content[idx_summary + 1:idx_filters]
            particle_filters = content[idx_filters + 1:idx_stats]
            real_stats = content[max(0, idx_dist_header - 15):idx_dist_header]
            
            df_dist = pd.read_csv(io.StringIO("\n".join(content[idx_dist_header:])))[['Bin Center', 'Average']]
            df_dist = df_dist[~df_dist['Bin Center'].astype(str).str.contains('<|>')]
            df_dist['Bin Center'] = pd.to_numeric(df_dist['Bin Center'], errors='coerce')
            df_dist['Average'] = pd.to_numeric(df_dist['Average'], errors='coerce')

            buoyancy = extract_value(particle_filters, "Buoyancy", 1, 2).strip().lower()
            b_type = "NEG" if "neg" in buoyancy else "POS" if "pos" in buoyancy else "UNKNOWN"

            stats = {
                "Mean [nm]": str(int(round(float(extract_value(real_stats, "Mean [μm]", 1, 3))*1000))) if "N/A" not in extract_value(real_stats, "Mean [μm]", 1, 3) else "N/A",
                "Concentration [#/mL]": extract_value(real_stats, "Concentration [#/mL]", 1, 3),
                "Buoyancy": b_type
            }
            all_summaries[dataset_labels[uploaded_file.name]] = stats
            histogram_data.append((uploaded_file.name, df_dist, b_type))

        # --- 4. Appearance (Auto-Color Pairing) ---
        st.subheader("Step 3: Appearance Settings")
        colors = {"Red": "#d62728", "Blue": "#1f77b4", "Green": "#2ca02c", "Purple": "#9467bd", "Black": "#000000", "Orange": "#ff7f0e"}
        color_names = list(colors.keys())
        
        # Group pairs for shared colors
        unique_groups = []
        for filename, _, _ in histogram_data:
            base = re.sub(r'[\s_-]*\b(POS|NEG)\b[\s_-]*', '', dataset_labels[filename], flags=re.IGNORECASE).strip()
            if base not in unique_groups: unique_groups.append(base)

        set_colors, set_styles = {}, {}
        for filename, _, _ in histogram_data:
            label = dataset_labels[filename]
            base = re.sub(r'[\s_-]*\b(POS|NEG)\b[\s_-]*', '', label, flags=re.IGNORECASE).strip()
            idx = unique_groups.index(base)
            
            c_col, s_col = st.columns(2)
            with c_col:
                sel_c = st.selectbox(f"Color: {label}", color_names, index=idx % len(color_names), key=f"c_{filename}")
                set_colors[filename] = colors[sel_c]
            with s_col:
                set_styles[filename] = st.selectbox(f"Style: {label}", ["-", "--", ":", "-."], key=f"s_{filename}")

        # --- 5. Plotting ---
        plot_title = st.text_input("Plot Title", value="Mirrored Buoyancy Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for fname, df, b_type in histogram_data:
            df = df.dropna()
            x = df["Bin Center"] * 1000
            if b_type == "NEG": x = -x
            ax.plot(x, df["Average"], label=dataset_labels[fname], color=set_colors[fname], linestyle=set_styles[fname], linewidth=2)

        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Diameter [nm]")
        ax.set_ylabel("Concentration [#/mL]")
        ax.set_title(plot_title, fontweight="bold")
        
        # Mirror Logic
        xlim = ax.get_xlim()
        max_x = max(abs(xlim[0]), abs(xlim[1]))
        ax.set_xlim(-max_x, max_x)
        ax.set_xticklabels([str(abs(int(t))) for t in ax.get_xticks()])
        ax.legend()
        
        st.pyplot(fig)
        
        # Table
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(all_summaries).T, use_container_width=True)
