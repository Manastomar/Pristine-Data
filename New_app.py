import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Function to display summary of data types
def summary(df):
    if df is not None:
        dt = pd.DataFrame(df.dtypes).reset_index()
        dt.columns = ['Column', 'Data Type']
        return dt

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to provide a download link for the CSV
def get_download_link(df, filename="updated_data_web_app.csv"):
    csv = convert_df_to_csv(df)
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

# Function to render interactive Matplotlib visualization module
def render_visualization_module(df):
    st.write("---")
    st.header("📊 Data Visualization (Matplotlib)")
    st.write("> Create insightful charts and graphs from your cleaned data.")

    if df is None or df.empty:
        st.warning("⚠️ No data available to visualize. Please upload and clean a CSV file.")
        return

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()

    with st.container():
        col_type, col_theme = st.columns([2, 1])

        with col_type:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart"],
                key="viz_chart_type"
            )

        with col_theme:
            style_options = ["default", "seaborn-v0_8-whitegrid", "ggplot", "fivethirtyeight", "bmh", "dark_background"]
            available_styles = [s for s in style_options if s in plt.style.available] or ["default"]
            selected_style = st.selectbox("Plot Theme / Style", options=available_styles, index=0, key="viz_theme")

        # Customization settings expander
        with st.expander("🎨 Appearance & Canvas Settings", expanded=False):
            c_title, c_rot = st.columns(2)
            with c_title:
                custom_title = st.text_input("Custom Chart Title", value="", placeholder="Leave blank for automatic title", key="viz_title")
            with c_rot:
                default_rot = 45 if chart_type in ["Bar Chart", "Line Chart", "Box Plot"] else 0
                x_rot = st.slider("X-axis Label Rotation (°)", min_value=0, max_value=90, value=default_rot, step=15, key="viz_rot")

            c_xlab, c_ylab = st.columns(2)
            with c_xlab:
                custom_xlabel = st.text_input("Custom X-axis Label", value="", placeholder="Leave blank for automatic label", key="viz_xlab")
            with c_ylab:
                custom_ylabel = st.text_input("Custom Y-axis Label", value="", placeholder="Leave blank for automatic label", key="viz_ylab")

            c_w, c_h, c_grid = st.columns([1, 1, 1])
            with c_w:
                fig_width = st.slider("Figure Width", min_value=6, max_value=16, value=10, key="viz_width")
            with c_h:
                fig_height = st.slider("Figure Height", min_value=3, max_value=12, value=5, key="viz_height")
            with c_grid:
                show_grid = st.checkbox("Show Grid Lines", value=True, key="viz_grid")

        # Plot generation
        valid_style = selected_style if selected_style in plt.style.available else "default"

        with plt.style.context(valid_style):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plot_ready = False

            # ----------------- 1. BAR CHART -----------------
            if chart_type == "Bar Chart":
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    x_col = st.selectbox("Select X-axis (Category / Grouping)", options=all_cols, key="bar_x_col")
                    bar_orient = st.radio("Bar Orientation", ["Vertical", "Horizontal"], horizontal=True, key="bar_orient")
                with b_c2:
                    y_mode_options = ["Count of Records (Frequency)"]
                    if numeric_cols:
                        y_mode_options.append("Numeric Column (Aggregated)")
                    y_mode = st.selectbox("Y-axis Metric Mode", y_mode_options, key="bar_y_mode")

                    if y_mode == "Numeric Column (Aggregated)":
                        y_col = st.selectbox("Select Y-axis (Numeric)", options=numeric_cols, key="bar_y_col")
                        agg_func = st.selectbox("Aggregation Method", ["Sum", "Mean", "Median", "Min", "Max", "None (Raw Data)"], key="bar_agg")

                b_c3, b_c4 = st.columns(2)
                with b_c3:
                    num_unique = df[x_col].nunique()
                    limit_bars = st.checkbox("Limit to Top N Categories", value=(num_unique > 15), key="bar_limit")
                    if limit_bars:
                        top_n = st.slider("Max Categories to Show", min_value=3, max_value=50, value=min(15, max(3, num_unique)), key="bar_top_n")
                with b_c4:
                    bar_color = st.color_picker("Bar Color", value="#2b5c8f", key="bar_color")
                    show_bar_labels = st.checkbox("Show Value Labels on Bars", value=True, key="bar_show_labels")

                try:
                    if y_mode == "Count of Records (Frequency)":
                        counts = df[x_col].astype(str).value_counts().dropna()
                        if limit_bars:
                            counts = counts.head(top_n)
                        x_labels = [str(x) for x in counts.index]
                        y_values = counts.values.tolist()
                        default_ylabel = "Count"
                        default_xlabel = str(x_col)
                        default_title = f"Frequency of {x_col}"
                    else:
                        if agg_func == "None (Raw Data)":
                            sub_df = df[[x_col, y_col]].dropna()
                            if limit_bars:
                                sub_df = sub_df.head(top_n)
                            x_labels = [str(x) for x in sub_df[x_col]]
                            y_values = sub_df[y_col].values.tolist()
                            default_ylabel = str(y_col)
                            default_title = f"{y_col} by {x_col} (Raw)"
                        else:
                            grouped = df.groupby(x_col)[y_col].agg(agg_func.lower()).dropna()
                            grouped = grouped.sort_values(ascending=False)
                            if limit_bars:
                                grouped = grouped.head(top_n)
                            x_labels = [str(x) for x in grouped.index]
                            y_values = grouped.values.tolist()
                            default_ylabel = f"{agg_func} of {y_col}"
                            default_title = f"{agg_func} of {y_col} by {x_col}"
                        default_xlabel = str(x_col)

                    if len(x_labels) == 0:
                        st.warning("No data found for the selected columns.")
                    else:
                        is_float = any(isinstance(v, (float, np.floating)) and not float(v).is_integer() for v in y_values)
                        label_fmt = '%.2f' if is_float else '%g'

                        if bar_orient == "Vertical":
                            bars = ax.bar(x_labels, y_values, color=bar_color, edgecolor='black', linewidth=0.6, alpha=0.85)
                            if show_bar_labels:
                                ax.bar_label(bars, padding=3, fmt=label_fmt)
                            ax.set_xticks(range(len(x_labels)))
                            ax.set_xticklabels(x_labels, rotation=x_rot, ha='right' if x_rot > 0 else 'center')
                            ax.set_xlabel(custom_xlabel or default_xlabel)
                            ax.set_ylabel(custom_ylabel or default_ylabel)
                        else:
                            # Horizontal bar: reverse so highest value is at the top
                            rev_labels = x_labels[::-1]
                            rev_values = y_values[::-1]
                            bars = ax.barh(rev_labels, rev_values, color=bar_color, edgecolor='black', linewidth=0.6, alpha=0.85)
                            if show_bar_labels:
                                ax.bar_label(bars, padding=3, fmt=label_fmt)
                            ax.set_xlabel(custom_ylabel or default_ylabel)
                            ax.set_ylabel(custom_xlabel or default_xlabel)

                        ax.set_title(custom_title or default_title, fontsize=13, fontweight='bold')
                        plot_ready = True
                except Exception as err:
                    st.error(f"Error generating Bar Chart: {err}")

            # ----------------- 2. LINE CHART -----------------
            elif chart_type == "Line Chart":
                if not numeric_cols:
                    st.warning("⚠️ Line chart requires at least one numeric column for the Y-axis.")
                    plt.close(fig)
                    return

                l_c1, l_c2 = st.columns(2)
                with l_c1:
                    x_options = ["Index (Row Number)"] + all_cols
                    x_col = st.selectbox("Select X-axis", options=x_options, key="line_x_col")
                    sort_x = st.checkbox("Sort along X-axis", value=True, key="line_sort_x")
                with l_c2:
                    y_cols = st.multiselect("Select Y-axis Column(s)", options=numeric_cols, default=[numeric_cols[0]], key="line_y_cols")

                l_c3, l_c4 = st.columns(2)
                with l_c3:
                    marker_style = st.selectbox("Marker Style", ["None", "Circle (o)", "Square (s)", "Triangle (^)", "Diamond (d)"], key="line_marker")
                    line_style = st.selectbox("Line Style", ["Solid (-)", "Dashed (--)", "Dotted (:)", "Dash-dot (-.)"], key="line_style")
                with l_c4:
                    line_width = st.slider("Line Width", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="line_width")
                    fill_area = st.checkbox("Shade Area Under Line", value=False, key="line_fill")

                if not y_cols:
                    st.warning("Please select at least one numeric column to plot on the Y-axis.")
                else:
                    try:
                        marker_map = {"None": None, "Circle (o)": "o", "Square (s)": "s", "Triangle (^)": "^", "Diamond (d)": "d"}
                        style_map = {"Solid (-)": "-", "Dashed (--)": "--", "Dotted (:)": ":", "Dash-dot (-.)": "-."}

                        if x_col == "Index (Row Number)":
                            sub_df = df[y_cols].copy().dropna(how='all')
                            x_vals = sub_df.index
                            default_xlabel = "Row Index"
                        else:
                            sub_df = df[[x_col] + y_cols].copy().dropna(subset=[x_col])
                            if sort_x:
                                sub_df = sub_df.sort_values(by=x_col)
                            x_vals = sub_df[x_col]
                            default_xlabel = str(x_col)

                        for col_name in y_cols:
                            ax.plot(
                                x_vals,
                                sub_df[col_name],
                                label=str(col_name),
                                marker=marker_map[marker_style],
                                linestyle=style_map[line_style],
                                linewidth=line_width,
                                alpha=0.9
                            )
                            if fill_area:
                                ax.fill_between(x_vals, sub_df[col_name], alpha=0.15)

                        ax.set_xlabel(custom_xlabel or default_xlabel)
                        default_ylabel = ", ".join(y_cols) if len(y_cols) <= 2 else "Values"
                        ax.set_ylabel(custom_ylabel or default_ylabel)
                        default_title = f"Line Chart: {default_ylabel} vs {default_xlabel}"
                        ax.set_title(custom_title or default_title, fontsize=13, fontweight='bold')
                        ax.legend(loc='best')

                        if x_rot > 0:
                            ax.tick_params(axis='x', rotation=x_rot)

                        plot_ready = True
                    except Exception as err:
                        st.error(f"Error generating Line Chart: {err}")

            # Display and Export
            if plot_ready:
                if show_grid:
                    ax.grid(True, linestyle='--', alpha=0.6)
                fig.tight_layout()

                st.pyplot(fig)

                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
                img_buffer.seek(0)
                st.download_button(
                    label="📥 Download Chart (High-Res PNG)",
                    data=img_buffer,
                    file_name=f"{chart_type.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    key="btn_download_chart"
                )

            plt.close(fig)


# Button CSS for uniform styling
st.markdown("""
<style>
.stButton > button {
    width: 150px;
    height: 50px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Function to reset session state
def reset_session_state():
    st.session_state.clear()

# Sidebar for file upload and reset button
st.sidebar.header('📤 Upload your CSV file here')
file = st.sidebar.file_uploader("", type="csv")

st.sidebar.write("> Click the Reset button to undo all changes")
if st.sidebar.button('Reset'):
    reset_session_state()

# Check if a new file is uploaded and reset session state if needed
if file is not None and 'last_uploaded_file' in st.session_state:
    if st.session_state.last_uploaded_file != file:
        reset_session_state()

# Main app title and instructions
st.title("Welcome to Data Cleaning App")
st.write("> Please use the sidebar on your left to upload files")

# Main app logic
if file is None:
    st.warning('Please upload a file.')
else:
    try:
        # Save the file name in session to detect new file uploads
        st.session_state.last_uploaded_file = file

        # Read CSV file
        df = pd.read_csv(file)

        # Initialize session state for modified columns if not already initialized
        if 'modified_columns' not in st.session_state:
            st.session_state.modified_columns = df.columns.tolist()

        st.write('---')
        st.title('File Overview')

        # Display the summary of the uploaded file
        col1, col2 = st.columns(2)

        with col1:
            with st.expander('Data Overview', expanded=True):
                shape = df.shape
                st.write(f'Columns = {shape[1]}')
                st.write(f'Rows = {shape[0]}')
                st.write(summary(df))

        with col2:
            st.write('## Data Type Conversion')
            change_dtype = st.multiselect("Select columns to change data type", options=df.columns)
            for column in change_dtype:
                current_type = df[column].dtype
                col_type = st.selectbox(
                    f"Select data type for {column} (current: {current_type})",
                    ["Keep current", "numeric", "datetime", "categorical"]
                )
                if col_type == "numeric":
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif col_type == "datetime":
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif col_type == "categorical":
                    df[column] = df[column].astype('category')
            st.success('Changes reflected in Data Overview')

        st.write("*")
        st.write('### Column Name Changes')
        with st.expander("Change Column Names", expanded=True):
            before_col = st.session_state.modified_columns
            before_col_df = pd.DataFrame(before_col, columns=['Column Name'])
            st.table(before_col_df)

            col3, col4, col5, col6 = st.columns(4)
            if st.button('Capital'):
                st.session_state.modified_columns = [col.title() for col in before_col]
            if st.button('Uppercase'):
                st.session_state.modified_columns = [col.upper() for col in before_col]
            if st.button('Lowercase'):
                st.session_state.modified_columns = [col.lower() for col in before_col]
            if st.button('Spaces ➡ Underscore'):
                st.session_state.modified_columns = [col.replace(" ", "_") for col in before_col]
            df.columns = st.session_state.modified_columns

            st.success("Column name changes applied.")
            st.table(pd.DataFrame(df.columns, columns=['Updated Column Names']))

        # Rename column functionality
        st.write("### Rename Column Names")
        column_select = st.selectbox('Select column to rename', options=st.session_state.modified_columns)
        new_column_name = st.text_input('Enter new column name')
        if st.button('Update Column Name'):
            if column_select and new_column_name:
                st.session_state.modified_columns = [
                    new_column_name if col == column_select else col
                    for col in st.session_state.modified_columns
                ]
                df.columns = st.session_state.modified_columns
                st.success("Column name updated.")
                st.table(pd.DataFrame(df.columns, columns=['Updated Column Names']))

        # Handle duplicates
        st.write("### Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        st.write(f"No. of Duplicate rows: {duplicate_count}")
        if st.button('Delete Duplicates'):
            df.drop_duplicates(inplace=True)
            st.success('Successfully deleted duplicates')

        # Handle missing values
        st.write('### Missing Values')
        missing_summary = df.isnull().sum()
        st.write("Columns with missing values:")
        st.write(missing_summary[missing_summary > 0])

        if st.button('Delete Columns with Missing Values'):
            df.dropna(axis=1, inplace=True)
            st.success('Column-wise deletion successful')
        if st.button('Delete Rows with Missing Values'):
            df.dropna(axis=0, inplace=True)
            st.success('Row-wise deletion successful')

        # Filling missing values
        st.write('### Filling Missing Values')
        fill_columns = st.multiselect('Select columns to fill missing values', options=missing_summary[missing_summary > 0].index)
        for column in fill_columns:
            fill_option = st.selectbox(f'Fill missing values in {column} with:', ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill"])
            if fill_option == "Mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif fill_option == "Median":
                df[column].fillna(df[column].median(), inplace=True)
            elif fill_option == "Mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif fill_option == "Zero":
                df[column].fillna(0, inplace=True)
            elif fill_option == "Forward Fill":
                df[column].fillna(method='ffill', inplace=True)
            elif fill_option == "Backward Fill":
                df[column].fillna(method='bfill', inplace=True)
            st.success(f'Missing values in {column} filled with {fill_option}.')

        # Updated data overview
        st.write("### Data Overview After Updates")
        st.table(df.head(8))

        # Data Visualization Module (Matplotlib: Bar, Line, Scatter, etc.)
        render_visualization_module(df)

        # Provide download link for the updated dataframe
        st.sidebar.write("## 🗂 Download Updated Data")
        get_download_link(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")