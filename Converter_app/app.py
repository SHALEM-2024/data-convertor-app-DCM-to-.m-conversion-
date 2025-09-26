# app.py
import io
import re  # ⬅️ add
import pandas as pd
import streamlit as st

from kennfeld_tools import (
    load_kennfeld_text,
    parse_all_kennfelds,
    build_excel_bytes,
    build_matlab_m_bytes,
)

st.set_page_config(page_title="KENNFELD → Excel/CSV", page_icon="", layout="wide")

# Widen the sidebar a bit
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { min-width: 480px; max-width: 0px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("DCM/TXT → Excel & MATLAB (.m)")
st.caption(
    "Upload a .DCM or .TXT with KENNFELD / GRUPPENKENNFELD / KENNLINIE / FESTWERT blocks. "
    "Optionally filter to specific map names."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Options")

    # Initialize a persistent pool of known targets
    DEFAULT_POOL = []
    if "target_pool" not in st.session_state:
        st.session_state.target_pool = DEFAULT_POOL.copy()

    # alias_map: original_name -> alias (if blank/missing, UI shows original)
    if "alias_map" not in st.session_state:
        st.session_state.alias_map = {name: name for name in st.session_state.target_pool}

    use_filter = st.checkbox("Filter to specific map names", value=True)

    # ---------- Bulk add/edit (Option A) ----------
    def _matlab_ident_preview(name: str) -> str:
        s = re.sub(r'[^A-Za-z0-9_]', '_', name.strip())
        if not s or not s[0].isalpha():
            s = 'm_' + s
        return s

    with st.expander("Bulk add/edit names & aliases", expanded=False):
        mode = st.radio("Mode", ["Merge", "Replace"], horizontal=True)
        pasted = st.text_area(
            "Paste mappings (one per line)",
            height=220,
            placeholder=(
                "DZWOLA = Idle_Warmup\n"
                "KFMSWDKQ -> DK_Q_Flow\n"
                "MDNORM\n"
                "KFRLSN, Knock_RL\n"
                "# Lines can use =  ->  ,  :  ;  or TAB. Blank lines and lines starting with # are ignored."
            ),
        )

        def _parse_bulk(text: str):
            entries, bad = [], []
            for raw in (text or "").splitlines():
                ln = raw.strip()
                if not ln or ln.startswith("#"):
                    continue
                # normalize fancy arrows/dashes
                ln = ln.replace("→", "->").replace("—", "-").replace("–", "-")
                parts = re.split(r"\s*(?:=|->|,|:|;|\t)\s*", ln, maxsplit=1)
                if len(parts) == 1:
                    name, alias = parts[0], parts[0]
                else:
                    name, alias = parts[0], parts[1]
                name, alias = (name or "").strip(), (alias or "").strip()
                if not name:
                    bad.append(raw); continue
                if not alias:
                    alias = name
                entries.append((name, alias))
            return entries, bad

        cols = st.columns([1,1,1])
        with cols[0]:
            apply_clicked = st.button("Parse & Apply", use_container_width=True)
        with cols[1]:
            preview_clicked = st.button("Preview", use_container_width=True)

        if preview_clicked and pasted:
            parsed, bad = _parse_bulk(pasted)
            if bad:
                st.warning(f"Ignored {len(bad)} invalid line(s).")
            if not parsed:
                st.info("Nothing parsed.")
            else:
                # Build a preview DF
                prev = []
                for name, alias in parsed:
                    prev.append({
                        "Original": name,
                        "Alias": alias,
                        "MATLAB Field (sanitized)": _matlab_ident_preview(alias),
                    })
                st.dataframe(pd.DataFrame(prev), use_container_width=True)

                # Warn on potential sanitized duplicates
                sani = [_matlab_ident_preview(a) for _, a in parsed]
                dups = {s for s in sani if sani.count(s) > 1}
                if dups:
                    st.warning(
                        "Multiple aliases sanitize to the same MATLAB field name: "
                        + ", ".join(sorted(dups))
                        + ". The exporter will de-duplicate by appending _2, _3, …"
                    )

        if apply_clicked and pasted:
            parsed, bad = _parse_bulk(pasted)
            if bad:
                st.warning(f"Ignored {len(bad)} invalid line(s).")

            added = updated = 0
            if mode == "Replace":
                new_pool: list[str] = []
                new_alias: dict[str, str] = {}
                for name, alias in parsed:
                    if name not in new_pool:
                        new_pool.append(name)
                    new_alias[name] = alias or name
                st.session_state.target_pool = new_pool
                st.session_state.alias_map = new_alias
                added = len(new_pool)
                updated = 0
            else:  # Merge
                for name, alias in parsed:
                    if name not in st.session_state.target_pool:
                        st.session_state.target_pool.append(name)
                        st.session_state.alias_map[name] = alias or name
                        added += 1
                    else:
                        if alias and st.session_state.alias_map.get(name) != alias:
                            st.session_state.alias_map[name] = alias
                            updated += 1

            # Quick dup check on sanitized fields (informational)
            sani = [_matlab_ident_preview(st.session_state.alias_map[n]) for n in st.session_state.target_pool]
            dups = {s for s in sani if sani.count(s) > 1}
            if dups:
                st.warning(
                    "Note: some aliases sanitize to duplicate MATLAB fields: "
                    + ", ".join(sorted(dups))
                    + ". The exporter will de-duplicate by appending _2, _3, …"
                )

            st.success(f"Bulk apply done. Added: {added}, Updated aliases: {updated}.")
            st.rerun()

    # ---------- Normal selector (shows alias) ----------
    selected_targets = st.multiselect(
        "Select data to extract",
        options=st.session_state.target_pool,
        default=st.session_state.target_pool,
        help="Choose which data to extract. Leave empty to include all data found.",
        format_func=lambda nm: st.session_state.alias_map.get(nm, nm) or nm,
    )

    # Optional single add UI (kept)
    if "show_add_box" not in st.session_state:
        st.session_state.show_add_box = False
    cols = st.columns([4, 4])
    with cols[0]:
        if st.button("Click to add"):
            st.session_state.show_add_box = True

    if st.session_state.show_add_box:
        with st.form("add_target_form", clear_on_submit=True):
            new_item = st.text_input("Add a map name", placeholder="e.g., NEW_MAP_NAME")
            new_alias = st.text_input("Alias (optional) — shown in UI and used in MATLAB", placeholder="e.g., Idle_Torque")
            add_submitted = st.form_submit_button("Add")
            cancel_clicked = st.form_submit_button("Cancel")
            if add_submitted:
                new_item = (new_item or "").strip()
                alias = (new_alias or "").strip()
                if new_item:
                    if new_item not in st.session_state.target_pool:
                        st.session_state.target_pool.append(new_item)
                        st.session_state.alias_map[new_item] = alias or new_item
                        shown = st.session_state.alias_map[new_item]
                        st.success(f"Added '{new_item}' (alias: '{shown}')")
                    else:
                        if alias:
                            st.session_state.alias_map[new_item] = alias
                            st.info(f"Updated alias for '{new_item}' → '{alias}'")
                        else:
                            st.info(f"'{new_item}' is already in the list.")
                st.session_state.show_add_box = False
                st.rerun()
            elif cancel_clicked:
                st.session_state.show_add_box = False
                st.rerun()

    round_digits = st.slider(
        "Round values (decimals)", 0, 8, 4,
        help="Applies to the Excel download (smaller file, cleaner look)."
    )

uploaded = st.file_uploader("Upload .DCM or .TXT", type=["dcm", "txt"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

# Build include_names from selection
include_names = None
if use_filter and selected_targets:
    include_names = selected_targets  # only extract these

# -------- Parse once, build both outputs --------
with st.spinner("Parsing data and preparing downloads…"):
    try:
        raw_bytes = uploaded.read()
        text = load_kennfeld_text(raw_bytes)
        data = parse_all_kennfelds(text, include_names=include_names)
        excel_bytes = build_excel_bytes(data, add_summary=True, round_digits=round_digits)
        # Use aliases for MATLAB struct field names & .name
        m_bytes = build_matlab_m_bytes(
            data,
            round_digits=round_digits,
            top_var="data",
            alias_map=st.session_state.alias_map,
        )
    except Exception as e:
        st.error(f"Failed to parse: {e}")
        st.stop()

st.success(f"Parsed **{len(data)}** map(s). Choose an export below:")

# Two download buttons side-by-side
b1, b2 = st.columns(2)
with b1:
    st.download_button(
        "⬇️ Download Excel (all data)",
        data=excel_bytes,
        file_name="KENNFELD_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with b2:
    st.download_button(
        "⬇️ Download MATLAB .m (all data)",
        data=m_bytes,
        file_name="kennfeld_data.m",
        mime="text/x-matlab",
        use_container_width=True,
    )

# -------- Summary table --------
summary_rows = []
for name, (_, _, _, df, meta) in data.items():
    summary_rows.append({
        "Map": name,
        "Alias": st.session_state.alias_map.get(name, name),
        "Kind": meta.get("kind", "KENNFELD"),
        "Dim": meta.get("dim", 2),
        "Rows (ny)": meta["ny"],
        "Cols (nx)": meta["nx"],
        "Unit X": meta["units"]["X"],
        "Unit Y": meta["units"]["Y"],
        "Unit W": meta["units"]["W"],
    })

st.subheader("Summary")
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# -------- Preview (table only) --------
st.subheader("Preview a map")
map_names = list(data.keys())
selected = st.selectbox(
    "Choose a map to preview",
    map_names,
    index=0,
    format_func=lambda nm: f"{st.session_state.alias_map.get(nm, nm)}  (orig: {nm})" if st.session_state.alias_map.get(nm, nm) != nm else nm
)

x, y, z, df, meta = data[selected]

st.markdown(
    f"**{st.session_state.alias_map.get(selected, selected)}** (orig: `{selected}`) — "
    f"kind: `{meta.get('kind', 'KENNFELD')}`, dim: `{meta.get('dim', 2)}` — "
    f"shape: {meta['ny']}×{meta['nx']}  |  "
    f"units: X={meta['units']['X']}, Y={meta['units']['Y']}, W={meta['units']['W']}"
)
st.dataframe(df, use_container_width=True, height=450)

# Per-map CSV download (keep original name in filename)
csv_bytes = df.to_csv().encode("utf-8")
st.download_button(
    f"⬇️ Download CSV ({selected})",
    data=csv_bytes,
    file_name=f"{selected}.csv",
    mime="text/csv",
    use_container_width=True,
)
