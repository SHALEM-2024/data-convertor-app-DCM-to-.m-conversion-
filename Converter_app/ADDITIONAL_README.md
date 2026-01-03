# AutoMap Pro: ECU Calibration Data Transformer

A high-performance conversion suite designed to bridge the gap between automotive ECU calibration files (.DCM/.TXT) and modern analytical environments like MATLAB and Excel.

## 🛠 How It Works
1. **Extraction:** The engine uses advanced RegEx to scan text files for `KENNFELD`, `KENNLINIE`, and `FESTWERT` blocks.
2. **Sanitization:** It automatically strips DCM-specific comments (e.g., `/* ... */`, `//`, `*`) and normalizes whitespace.
3. **Transformation:** Data is converted into structured Pandas DataFrames for live preview and manipulation.
4. **Serialization:** Exports are generated as multi-sheet Excel workbooks or sanitized MATLAB scripts with unique variable identifiers.

## 🚀 Key Technical Highlights
* **Portable Deployment:** Includes a VBScript/Batch wrapper that automatically manages a Python virtual environment and handles dependency installation for non-technical users.
* **Dynamic Mapping:** Features a "Bulk Add/Edit" interface in Streamlit to map cryptic HEX labels to human-readable aliases.
* **Robust Encoding:** Handles UTF-8-sig, CP1252, and Latin-1 to ensure global compatibility with various calibration tools.

## 📦 Dependencies
* Python 3.x
* Streamlit, Pandas, NumPy, XlsxWriter
