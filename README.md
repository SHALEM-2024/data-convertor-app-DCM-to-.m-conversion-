# DCM/TXT to Excel & MATLAB Data Converter

A specialized engineering tool built with Python and Streamlit to parse, visualize, and convert automotive ECU calibration files (.DCM and .TXT) into accessible formats for analysis and simulation.

## 🚀 Features
* [cite_start]**Multi-Format Support:** Parses `KENNFELD` (2D), `KENNLINIE` (1D), and `FESTWERT` (Scalar) data blocks[cite: 1, 3].
* [cite_start]**Intelligent Data Cleaning:** Automatically handles DCM-specific comments, exotic whitespace, and multiple encodings (UTF-8, CP1252, Latin-1)[cite: 3].
* [cite_start]**Dynamic Alias Mapping:** Bulk-rename cryptic ECU labels to human-readable names via a regex-powered mapping interface[cite: 1].
* [cite_start]**Professional Exporting:** * **Excel:** Multi-sheet workbooks with data rounding and summary tables[cite: 3].
    * [cite_start]**MATLAB:** Generates sanitized `.m` scripts with unique variable identifiers for direct use in simulations[cite: 3].
* [cite_start]**Interactive Visualization:** Live data previews using Streamlit and Pandas before exporting[cite: 1].

## 🛠️ Tech Stack
* [cite_start]**Language:** Python 3.x [cite: 4]
* [cite_start]**Frontend:** Streamlit [cite: 1]
* [cite_start]**Data Processing:** Pandas, NumPy [cite: 3]
* [cite_start]**File Handling:** XlsxWriter [cite: 3]
* [cite_start]**Automation:** VBScript and Batch for portable execution [cite: 2, 4]

## 📥 Installation & Usage
1. Clone this repository.
2. Ensure you have Python installed.
3. [cite_start]**Windows Users:** Simply double-click `start_app.vbs` to automatically set up the virtual environment, install dependencies, and launch the app[cite: 2, 4].
4. **Manual Run:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
