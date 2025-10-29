import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
import io
import unicodedata
from matplotlib import pyplot as plt
from matplotlib.table import Table
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

        # Summary Table
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
            "Experiment Duration [mm:ss]": convert_seconds_to_min_sec(extract_value(summary_data, "Experiment Duration [s]", 1, 2)),
            "Buoyancy": extract_value(particle_filters, "Buoyancy", 1, 2),
            "# Particles After Filtering": extract_value(particle_filters, "# Particles After Filtering", 1, 2),
            "# Particles Measured": extract_value(summary_data, "# Particles Measured", 1, 2),
            "# Particles Detected": extract_value(summary_data, "# Particles Detected", 1, 2),
            "Coincidence (%)": extract_value(summary_data, "Coincidence [%]", 1, 2),
            "Limit of Detection [μm]": extract_value(content, "Limit of Detection [μm]", 1, 2)
        }

        # Determine buoyancy from metadata (Positive or Negative)
        buoyancy = summary_table["Buoyancy"].strip().lower()
        if "neg" in buoyancy:
            buoyancy_type = "NEG"
        elif "pos" in buoyancy:
            buoyancy_type = "POS"
        else:
            buoyancy_type = "UNKNOWN"

        summary_table["Buoyancy Type"] = buoyancy_type
        all_summaries[dataset_labels[filename]] = summary_table

        # Store data
        histogram_data.append((filename, df_dist, buoyancy_type))

    # --- Color selection ---
    st.subheader("Choose Colors for Each Dataset")
    generic_colors = {
        "Black": "#000000", "Blue": "#1f77b4", "Red": "#d62728",
        "Green": "#2ca02c", "Orange": "#ff7f0e", "Purple": "#9467bd",
        "Brown": "#8c564b", "Pink": "#e377c2", "Olive": "#bcbd22",
        "Cyan": "#17becf", "Gray": "#7f7f7f"
    }
    generic_color_names = list(generic_colors.keys())
    default_cycle = ["Black", "Red", "Blue", "Green"]

    dataset_colors = {}
    for i, (filename, _, _) in enumerate(histogram_data):
        label = dataset_labels[filename]
        default_color_name = default_cycle[i % len(default_cycle)]
        selected_color_name = st.selectbox(
            f"Color for {label}",
            generic_color_names,
            index=generic_color_names.index(default_color_name),
            key=f"color_{label}"
        )
        dataset_colors[filename] = generic_colors[selected_color_name]

    # --- Marker and line style options ---
    st.subheader("Choose Marker Shape and Line Style")
    marker_options = {"None": None, "Circle": 'o', "Triangle": '^', "Square": 's', "Diamond": 'D', "Star": '*', "X": 'x', "Plus": '+'}
    line_style_options = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
    dataset_markers, dataset_marker_sizes, dataset_line_styles, dataset_line_widths = {}, {}, {}, {}

    for filename, _, _ in histogram_data:
        label = dataset_labels[filename]
        dataset_markers[filename] = marker_options[st.selectbox(f"Marker for {label}", list(marker_options.keys()), index=0, key=f"marker_{label}")]
        dataset_marker_sizes[filename] = st.slider(f"Marker size for {label}", 4, 20, 8, key=f"marker_size_{label}")
        dataset_line_styles[filename] = line_style_options[st.selectbox(f"Line style for {label}", list(line_style_options.keys()), index=0, key=f"linestyle_{label}")]
        dataset_line_widths[filename] = st.slider(f"Line width for {label}", 1, 6, 2, key=f"linewidth_{label}")

    plot_title = st.text_input("Enter a title for the mirrored buoyancy plot:", value="Mirrored Buoyancy Distribution")

    # --- Mirrored Plot ---
    fig, ax = plt.subplots()
    for filename, df, buoyancy_type in histogram_data:
        df_clean = df.dropna(subset=["Bin Center", "Average"])
        x = df_clean["Bin Center"]
        y = df_clean["Average"]

        if buoyancy_type == "NEG":
            x = -x  # Mirror left
            label_suffix = " (NEG)"
        elif buoyancy_type == "POS":
            label_suffix = " (POS)"
        else:
            label_suffix = ""

        ax.plot(
            x, y,
            label=f"{dataset_labels[filename]}{label_suffix}",
            color=dataset_colors[filename],
            linestyle=dataset_line_styles[filename],
            linewidth=dataset_line_widths[filename],
            marker=dataset_markers[filename],
            markersize=dataset_marker_sizes[filename]
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Diameter [μm] (Negative = Negatively Buoyant, Positive = Positively Buoyant)")
    ax.set_ylabel("Concentration [#/mL]")
    ax.set_title(plot_title)
    ax.legend(title="Datasets")
    st.pyplot(fig)

    # --- Summary Table ---
    combined_summary = pd.DataFrame(all_summaries)
    st.subheader("Summary Table Comparison")
    st.dataframe(combined_summary, use_container_width=True)
    # --- Download Mirrored Plot as SVG ---
    svg_buffer = BytesIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_data = svg_buffer.getvalue()
    b64_svg = base64.b64encode(svg_data).decode("utf-8")
    href_svg = f'<a href="data:image/svg+xml;base64,{b64_svg}" download="mirrored_buoyancy_plot.svg">📥 Download Mirrored Plot (SVG)</a>'
    st.markdown(href_svg, unsafe_allow_html=True)

    # --- Download Mirrored Plot as PNG ---
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", bbox_inches="tight", dpi=300)
    png_data = png_buffer.getvalue()
    b64_png = base64.b64encode(png_data).decode("utf-8")
    href_png = f'<a href="data:image/png;base64,{b64_png}" download="mirrored_buoyancy_plot.png">📸 Download Mirrored Plot (PNG)</a>'
    st.markdown(href_png, unsafe_allow_html=True)

    # --- Download Summary Table as CSV ---
    csv_buffer = StringIO()
    combined_summary.to_csv(csv_buffer)
    b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode("utf-8")
    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="summary_table.csv">📥 Download Summary Table (CSV)</a>'
    st.markdown(href_csv, unsafe_allow_html=True)

    # --- Render Summary Table as Figure (SVG + PNG) ---
    summary_fig = render_table_as_figure(combined_summary)
    svg_table_buffer = BytesIO()
    summary_fig.savefig(svg_table_buffer, format="svg", bbox_inches="tight")
    svg_table_data = svg_table_buffer.getvalue()
    b64_table_svg = base64.b64encode(svg_table_data).decode("utf-8")

    st.markdown(
        f'<a href="data:image/svg+xml;base64,{b64_table_svg}" download="summary_table.svg">📥 Download Summary Table (SVG)</a>',
        unsafe_allow_html=True
    )

    # --- Download Summary Table as PNG ---
    png_table_buffer = BytesIO()
    summary_fig.savefig(png_table_buffer, format="png", bbox_inches="tight", dpi=200)
    png_table_data = png_table_buffer.getvalue()
    b64_table_png = base64.b64encode(png_table_data).decode("utf-8")

    st.markdown(
        f'<a href="data:image/png;base64,{b64_table_png}" download="summary_table.png">📸 Download Summary Table (PNG)</a>',
        unsafe_allow_html=True
    )
