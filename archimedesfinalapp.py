import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
import io
import unicodedata
from matplotlib import pyplot as plt
from matplotlib.table import Table


st.set_page_config(layout="wide")

st.title("Archimedes to CSV")

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
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8").strip().lower()

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
uploaded_files = st.file_uploader("Upload one or more CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_summaries = {}
    histogram_data = []
    dataset_labels = {}

    st.subheader("Customize Dataset Names")
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        default_label = filename.rsplit(".", 1)[0]
        user_label = st.text_input(f"Label for {filename}", value=default_label)
        dataset_labels[filename] = user_label

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        content = uploaded_file.read().decode("utf-8").splitlines()

        # Section indices
        idx_summary = find_index(content, "SUMMARY DATA")
        idx_filters = find_index(content, "PARTICLE FILTERS")
        idx_stats = find_index(content, "SUMMARY STATISTICS")
        idx_dist_header = find_index(content, "Bin Start")

        # Sections
        summary_data = content[idx_summary + 1:idx_filters]
        particle_filters = content[idx_filters + 1:idx_stats]
        real_stats = content[max(0, idx_dist_header - 15):idx_dist_header]
        particle_distribution_lines = content[idx_dist_header:]

        # Histogram Data
        df_dist = pd.read_csv(io.StringIO("\n".join(particle_distribution_lines)))
        df_dist = df_dist[['Bin Center', 'Average']].copy()
        df_dist = df_dist[~df_dist['Bin Center'].astype(str).str.contains('<|>')]
        df_dist['Bin Center'] = pd.to_numeric(df_dist['Bin Center'], errors='coerce')
        df_dist['Average'] = pd.to_numeric(df_dist['Average'], errors='coerce')
        histogram_data.append((filename, df_dist))

        # Summary Table with µm-to-nm conversion
        def convert_um_to_nm(value):
            try:
                return str(int(round(float(value) * 1000)))
            except:
                return "N/A"

        def convert_seconds_to_min_sec(value):
            try:
                total_seconds = int(float(value))
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes:02}:{seconds:02}"
            except:
                return "N/A"


        summary_table = {
            "Mean [nm]": convert_um_to_nm(extract_value(real_stats, "Mean [μm]", 1, 3)),
            "Stdev [nm]": convert_um_to_nm(extract_value(real_stats, "Stdev [μm]", 1, 3)),
            "Mode [nm]": convert_um_to_nm(extract_value(real_stats, "Mode [μm]", 1, 3)),
            "Polydispersity": extract_value(real_stats, "Polydispersity", 1, 3),
            "Standard Error [μm]": extract_value(real_stats, "Standard Error [μm]", 1, 3),
            "Concentration [#/mL]": extract_value(real_stats, "Concentration [#/mL]", 1, 3),
            "# Particles After Filtering": extract_value(particle_filters, "# Particles After Filtering", 1, 2),
            "Buoyancy": extract_value(particle_filters, "Buoyancy", 1, 2),
            "Experiment Duration [mm:ss]": convert_seconds_to_min_sec(extract_value(summary_data, "Experiment Duration [s]", 1, 2)),
            "# Particles Measured": extract_value(summary_data, "# Particles Measured", 1, 2),
            "# Particles Detected": extract_value(summary_data, "# Particles Detected", 1, 2),
            "Limit of Detection [μm]": extract_value(summary_data, "Limit of Detection [μm]", 1, 2)
        }

        all_summaries[dataset_labels[filename]] = summary_table

    # Let user choose from predefined color options
    st.subheader("Choose Colors for Each Dataset")
    color_options = {
        "Black": "#000000",
        "Dark Gray": "#4D4D4D",
        "Gray": "#7f7f7f",
        "Light Gray": "#BFBFBF",
        "Blue": "#1f77b4",
        "Orange": "#ff7f0e",
        "Green": "#2ca02c",
        "Red": "#d62728",
        "Purple": "#9467bd",
        "Brown": "#8c564b",
        "Pink": "#e377c2",
        "Olive": "#bcbd22",
        "Cyan": "#17becf"
    }

    color_names = list(color_options.keys())

    dataset_colors = {}
    used_colors = set()

    for i, (filename, _) in enumerate(histogram_data):
        available_colors = [name for name in color_names if color_options[name] not in used_colors]
        default = available_colors[i % len(available_colors)]
        selected = st.selectbox(f"Color for {dataset_labels[filename]}", available_colors, index=available_colors.index(default))
        dataset_colors[filename] = color_options[selected]
        used_colors.add(color_options[selected])

    # Let user choose marker shape
    st.subheader("Choose Marker Shape for Each Dataset")
    marker_options = {
        "None": None,
        "Circle": 'o',
        "Triangle": '^',
        "Square": 's',
        "Diamond": 'D',
        "Star": '*',
        "X": 'x',
        "Plus": '+'
    }
    dataset_markers = {}
    dataset_marker_sizes = {}

    for i, (filename, _) in enumerate(histogram_data):
        label = dataset_labels[filename]
        selected_shape = st.selectbox(f"Marker for {label}", list(marker_options.keys()), index=i % len(marker_options))
        dataset_markers[filename] = marker_options[selected_shape]

        selected_size = st.slider(f"Marker size for {label}", min_value=4, max_value=20, value=8)
        dataset_marker_sizes[filename] = selected_size

    st.subheader("Set Plot Title")
    plot_title = st.text_input("Enter a title for the histogram:", value="Overlaid Histogram with Touching Bars and Markers")

    # --- Overlapping histograms with shape overlays ---
    st.subheader("Overlapping Particle Size Distributions (Overlaid Histogram)")

    def extract_bin_size(lines):
        for line in lines:
            if "Bin Size" in line:
                parts = line.split(',')
                for p in parts:
                    try:
                        return float(p)
                    except:
                        continue
        return 0.01

    bin_size = extract_bin_size(content)
    bar_width = bin_size * 0.95

    fig, ax = plt.subplots()

    for i, (filename, df) in enumerate(histogram_data):
        df_clean = df[~df['Bin Center'].astype(str).str.contains('<|>')]
        df_clean = df_clean[['Bin Center', 'Average']].dropna()
        df_clean['Bin Center'] = pd.to_numeric(df_clean['Bin Center'], errors='coerce')
        df_clean['Average'] = pd.to_numeric(df_clean['Average'], errors='coerce')

        # Plot bars
        ax.bar(
            df_clean['Bin Center'],
            df_clean['Average'],
            width=bar_width,
            label=dataset_labels[filename],
            alpha=0.5,
            align='center',
            color=dataset_colors[filename]
        )
        # Overlay markers if selected
        marker_shape = dataset_markers[filename]
        if marker_shape:  # None or empty string means skip
            ax.plot(
                df_clean['Bin Center'],
                df_clean['Average'],
                linestyle='',
                marker=marker_shape,
                markersize=dataset_marker_sizes[filename],
                color=dataset_colors[filename],
                label="_nolegend_"
            )


    ax.set_xlabel("Diameter [μm]")
    ax.set_ylabel("Concentration [#/mL]")
    ax.set_title(plot_title)

    from matplotlib.lines import Line2D
    custom_handles = []
    for filename in dataset_labels:
        label = dataset_labels[filename]
        marker = dataset_markers[filename]
        color = dataset_colors[filename]
        size = dataset_marker_sizes[filename]
        handle = Line2D(
            [0], [0],
            marker=marker,
            color='w',
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=size,
            label=label,
            linestyle='None'
        )
        custom_handles.append(handle)

    ax.legend(handles=custom_handles, title="Datasets")
    st.pyplot(fig)
    # --- Download Histogram as SVG ---
    svg_buffer = BytesIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_data = svg_buffer.getvalue()
    b64_svg = base64.b64encode(svg_data).decode("utf-8")
    href_svg = f'<a href="data:image/svg+xml;base64,{b64_svg}" download="histogram.svg">📥 Download Histogram (SVG)</a>'
    st.markdown(href_svg, unsafe_allow_html=True)

    # --- Combine Summary Tables Side-by-Side ---
    combined_summary = pd.DataFrame(all_summaries)
    st.subheader("Summary Table Comparison")
    st.dataframe(combined_summary, use_container_width=True)

    # --- Download Summary Table as CSV ---
    csv_buffer = StringIO()
    combined_summary.to_csv(csv_buffer)
    b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode("utf-8")
    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="summary_table.csv">📥 Download Summary Table (CSV)</a>'
    st.markdown(href_csv, unsafe_allow_html=True)

    # --- Render summary table as vector figure (SVG) ---
    summary_fig = render_table_as_figure(combined_summary)

    svg_table_buffer = BytesIO()
    summary_fig.savefig(svg_table_buffer, format="svg", bbox_inches="tight")
    svg_table_data = svg_table_buffer.getvalue()
    b64_table_svg = base64.b64encode(svg_table_data).decode("utf-8")

    st.markdown(
        f'<a href="data:image/svg+xml;base64,{b64_table_svg}" download="summary_table.svg">📥 Download Summary Table (SVG)</a>',
        unsafe_allow_html=True
    )
