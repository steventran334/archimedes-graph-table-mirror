import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
import io
import unicodedata
from matplotlib.table import Table
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import re  # Regex for grouping filenames
import numpy as np

st.set_page_config(layout="wide")

st.title("Archimedes to CSV + Mirrored Buoyancy Plot")

st.markdown("""
    <style>
        .main .block-container {
            max-width: 100% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .css-18e3th9 {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .dataframe th, .dataframe td {
            padding: 0.5rem 1rem;
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

def render_table_as_figure(df, title="Summary Table", col_width=3.0, row_height=0.625, font_size=14):
    fig, ax = plt.subplots(figsize=(col_width * (df.shape[1] + 1), row_height * (df.shape[0] + 1)))
    ax.axis('off')
    mpl_table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    mpl_table.scale(1.2, 1.2)
    return fig

# --- Upload multiple CSVs ---
raw_uploaded_files = st.file_uploader("Upload one or more CSV files", type="csv", accept_multiple_files=True)

if raw_uploaded_files:
    # --- FILE REORDERING FEATURE ---
    # Map filenames to file objects
    file_dict = {f.name: f for f in raw_uploaded_files}
    file_names = list(file_dict.keys())
    
    st.subheader("Step 1: Define File Order & Selection")
    ordered_filenames = st.multiselect(
        "Select and reorder files (drag and drop or click to add in order)",
        options=file_names,
        default=file_names,
        help="The order selected here determines the order in the legend and summary table."
    )
    
    # Create the final list based on user selection
    uploaded_files = [file_dict[name] for name in ordered_filenames]

    all_summaries = {}
    histogram_data = []
    dataset_labels = {}
    
    if uploaded_files:
        st.subheader("Step 2: Customize Dataset Names")
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            default_label = filename.rsplit(".", 1)[0]
            user_label = st.text_input(f"Label for {filename}", value=default_label, key=f"label_{filename}")
            dataset_labels[filename] = user_label

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            uploaded_file.seek(0) 
            content = uploaded_file.read().decode("utf-8").splitlines()

            idx_summary = find_index(content, "SUMMARY DATA")
            idx_filters = find_index(content, "PARTICLE FILTERS")
            idx_stats = find_index(content, "SUMMARY STATISTICS")
            idx_dist_header = find_index(content, "Bin Start")

            if idx_dist_header is None:
                st.error(f"Could not find 'Bin Start' in {filename}. Skipping.")
                continue

            summary_data = content[idx_summary + 1:idx_filters]
            particle_filters = content[idx_filters + 1:idx_stats]
            real_stats = content[max(0, idx_dist_header - 15):idx_dist_header]
            particle_distribution_lines = content[idx_dist_header:]

            df_dist = pd.read_csv(io.StringIO("\n".join(particle_distribution_lines)))
            df_dist = df_dist[['Bin Center', 'Average']].copy()
            df_dist = df_dist[~df_dist['Bin Center'].astype(str).str.contains('<|>')]
            df_dist['Bin Center'] = pd.to_numeric(df_dist['Bin Center'], errors='coerce')
            df_dist['Average'] = pd.to_numeric(df_dist['Average'], errors='coerce')

            def convert_um_to_nm(value):
                try: return str(int(round(float(value) * 1000)))
                except: return "N/A"

            def convert_seconds_to_min_sec(value):
                try:
                    total_seconds = int(float(value))
                    return f"{total_seconds // 60:02}:{total_seconds % 60:02}"
                except: return "N/A"

            summary_table = {
                "Mean [nm]": convert_um_to_nm(extract_value(real_stats, "Mean [μm]", 1, 3)),
                "Stdev [nm]": convert_um_to_nm(extract_value(real_stats, "Stdev [μm]", 1, 3)),
                "Mode [nm]": convert_um_to_nm(extract_value(real_stats, "Mode [μm]", 1, 3)),
                "Polydispersity": extract_value(real_stats, "Polydispersity", 1, 3),
                "Standard Error [μm]": extract_value(real_stats, "Standard Error [μm]", 1, 3),
                "Concentration [#/mL]": extract_value(real_stats, "Concentration [#/mL]", 1, 3),
                "Experiment Duration [mm:ss]": convert_seconds_to_min_sec(extract_value(summary_data, "Experiment Duration [s]", 1, 2)),
                "Buoyancy": extract_value(particle_filters, "Buoyancy", 1, 2),
                "# Particles After Filtering": extract_value(particle_filters, "# Particles After Filtering", 1, 2),
                "# Particles Measured": extract_value(summary_data, "# Particles Measured", 1, 2),
                "# Particles Detected": extract_value(summary_data, "# Particles Detected", 1, 2),
                "Coincidence (%)": extract_value(summary_data, "Coincidence [%]", 1, 2)
            }

            buoyancy = summary_table["Buoyancy"].strip().lower()
            buoyancy_type = "NEG" if "neg" in buoyancy else "POS" if "pos" in buoyancy else "UNKNOWN"
            summary_table["Buoyancy Type"] = buoyancy_type
            all_summaries[dataset_labels[filename]] = summary_table
            histogram_data.append((filename, df_dist, buoyancy_type))

        # --- Color selection ---
        st.subheader("Step 3: Appearance Settings")
        generic_colors = {"Red": "#d62728", "Blue": "#1f77b4", "Green": "#2ca02c", "Purple": "#9467bd", "Black": "#000000", "Orange": "#ff7f0e", "Brown": "#8c564b", "Pink": "#e377c2", "Olive": "#bcbd22", "Cyan": "#17becf", "Gray": "#7f7f7f"}
        generic_color_names = list(generic_colors.keys())
        default_cycle = ["Red", "Blue", "Green", "Purple", "Black"]

        dataset_colors, dataset_markers, dataset_marker_sizes, dataset_line_styles, dataset_line_widths = {}, {}, {}, {}, {}
        
        for i, (filename, _, _) in enumerate(histogram_data):
            label = dataset_labels[filename]
            col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns([2, 1, 1, 1, 1])
            
            with col_c1:
                selected_color = st.selectbox(f"Color: {label}", generic_color_names, index=i % len(default_cycle), key=f"c_{filename}")
                dataset_colors[filename] = generic_colors[selected_color]
            with col_c2:
                dataset_markers[filename] = st.selectbox(f"Marker: {label}", ["None", "o", "^", "s", "D", "*"], key=f"m_{filename}")
            with col_c3:
                dataset_marker_sizes[filename] = st.slider(f"Size: {label}", 4, 20, 8, key=f"ms_{filename}")
            with col_c4:
                dataset_line_styles[filename] = st.selectbox(f"Line: {label}", ["-", "--", ":", "-."], key=f"ls_{filename}")
            with col_c5:
                dataset_line_widths[filename] = st.slider(f"Width: {label}", 1, 6, 2, key=f"lw_{filename}")

        plot_title = st.text_input("Plot Title:", value="")
        
        # --- Plotting ---
        if histogram_data:
            fig, ax = plt.subplots(figsize=(10, 7))
            for filename, df, buoyancy_type in histogram_data:
                df_clean = df.dropna(subset=["Bin Center", "Average"])
                x = df_clean["Bin Center"] * 1000
                y = df_clean["Average"]
                if buoyancy_type == "NEG": x = -x

                ax.plot(x, y, label=dataset_labels[filename], color=dataset_colors[filename],
                        linestyle=dataset_line_styles[filename], linewidth=dataset_line_widths[filename],
                        marker=None if dataset_markers[filename]=="None" else dataset_markers[filename],
                        markersize=dataset_marker_sizes[filename])

            # Formatting
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Diameter [nm]", fontsize=12, labelpad=20)
            ax.set_ylabel("Concentration [#/mL]", fontsize=12)
            ax.set_title(plot_title, fontsize=14, weight="bold")
            
            # X-Axis Adjustment
            st.subheader("Step 4: Adjust X-Axis Range")
            xc1, xc2 = st.columns(2)
            with xc1: max_neg = st.number_input("Left Side Max (NEG)", 0, 10000, 1000, 100)
            with xc2: max_pos = st.number_input("Right Side Max (POS)", 100, 10000, 1000, 100)
            
            ax.set_xlim(-max_neg, max_pos)
            ticks = ax.get_xticks()
            ax.set_xticklabels([str(abs(int(t))) for t in ticks])
            ax.spines['bottom'].set_position(('data', 0))
            ax.legend(loc="upper right")
            
            st.pyplot(fig)

            # Summary Table
            combined_summary = pd.DataFrame(all_summaries)
            st.subheader("Summary Table")
            st.dataframe(combined_summary, use_container_width=True)
