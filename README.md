# CSV Data Cleaning & Visualization Tool

An interactive web application built with **Streamlit**, **Pandas**, and **Matplotlib** designed to clean, transform, and visualize CSV datasets.

---

## 🚀 Features

### 🧹 Data Cleaning & Preprocessing
* **File Overview**: Inspect total rows, columns, and data types automatically upon CSV upload.
* **Data Type Conversion**: Convert column types on the fly to `numeric`, `datetime`, or `categorical`.
* **Column Name Formatting**: One-click bulk transformations (Title Case, Uppercase, Lowercase, Spaces to Underscore).
* **Column Renaming**: Select any individual column and rename it.
* **Duplicate Removal**: Detect and delete duplicate rows with a single click.
* **Missing Value Handling**:
  * Drop columns or rows containing missing values.
  * Impute missing data with `Mean`, `Median`, `Mode`, `Zero`, `Forward Fill`, or `Backward Fill`.
* **Export Cleaned Data**: Instant download button for the updated CSV.

### 📊 Data Visualization (Matplotlib)
* **Bar Chart**:
  * Supports Vertical and Horizontal orientations.
  * Frequency counts or aggregate numeric metrics (`Sum`, `Mean`, `Median`, `Min`, `Max`, `Raw`).
  * Option to limit to Top N categories with dynamic value labels on bars.
  * Custom bar color picker.
* **Line Chart**:
  * Multi-series plotting (visualize multiple numeric columns simultaneously).
  * Sort along X-axis toggle.
  * Customizable markers, line styles, line widths, and shaded area fills.
* **Canvas & Styling Controls**:
  * Matplotlib styles (`seaborn-v0_8-whitegrid`, `ggplot`, `fivethirtyeight`, `bmh`, `dark_background`, `default`).
  * Editable chart title, custom axis labels, and X-axis tick rotation.
  * Figure width and height sizing controls.
  * High-resolution (300 DPI) PNG chart download.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Manastomar/<repository-name>.git
   cd <repository-name>
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run New_app.py
   ```
   *Or with python:*
   ```bash
   python -m streamlit run New_app.py
   ```

---

## 📦 Requirements

* Python 3.8+
* `streamlit`
* `pandas`
* `matplotlib`
* `numpy`
