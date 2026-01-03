# DCM/TXT to MATLAB & Excel Data Converter

A specialized engineering tool built with Python and Streamlit to parse, visualize, and convert automotive ECU calibration files (.DCM and .TXT) into accessible formats for analysis and simulation.

## 🚀 Features  
(Two wheeler ECU related data, people working in the relevant domain can identify)
* **Multi-Format Support:** Parses `KENNFELD` (2D), `KENNLINIE` (1D), and `FESTWERT` (Scalar) data blocks.
* **Intelligent Data Cleaning:** Automatically handles DCM-specific comments, exotic whitespace, and multiple encodings (UTF-8, CP1252, Latin-1).
* **Dynamic Alias Mapping:** Bulk-rename cryptic ECU labels to human-readable names via a regex-powered mapping interface.
* **Professional Exporting:** * **Excel:** Multi-sheet workbooks with data rounding and summary tables.
    * **MATLAB:** Generates sanitized `.m` scripts (Matlab format) with unique variable identifiers for direct use in simulations.
* **Interactive Visualization:** Live data previews using Streamlit and Pandas before exporting.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **File Handling:** XlsxWriter
* **Automation:** VBScript and Batch for portable execution

## 📥 Installation & Usage
1. Clone this repository.
2. Ensure you have Python installed.
3. **Windows Users:** Simply double-click `start_app.vbs` to automatically set up the virtual environment, install dependencies, and launch the app.
4. **Manual Run:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
