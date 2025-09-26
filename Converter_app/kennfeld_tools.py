# kennfeld_tools.py
from __future__ import annotations
import io
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any, Optional

import numpy as np
import pandas as pd

# ---------- Header detectors (now includes KENNLINIE + FESTWERT) ----------
_HEADER_START_RE = re.compile(r'^\s*(KENNFELD|GRUPPENKENNFELD|KENNLINIE|FESTWERT)\b', re.I)
_HEADER_PARSE_2D_RE = re.compile(r'^\s*(KENNFELD|GRUPPENKENNFELD)\s+(\S+)\s+(\d+)\s+(\d+)\b', re.I)
_HEADER_PARSE_1D_RE = re.compile(r'^\s*(KENNLINIE)\s+(\S+)\s+(\d+)\b', re.I)
# Accept FESTWERT with or without a trailing "1"
_HEADER_PARSE_SCALAR_RE = re.compile(r'^\s*(FESTWERT)\s+(\S+)(?:\s+(\d+))?\b', re.I)

# =========================================================
# DCM-aware loader
# =========================================================
def load_kennfeld_text(src: str | os.PathLike | bytes) -> str:
    """
    Load text from a path (.dcm/.txt) OR raw bytes/string.
    Tries encodings, strips DCM comments, normalizes whitespace.
    """
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        # If 'src' is already raw content (not a path), just clean it
        if isinstance(src, str) and (
            "KENNFELD" in src or "GRUPPENKENNFELD" in src or "KENNLINIE" in src or "FESTWERT" in src or "\n" in src
        ):
            return _normalize_text(_strip_dcm_comments(src))
        data = Path(src).read_bytes()

    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("latin-1", errors="replace")

    return _normalize_text(_strip_dcm_comments(text))

def _strip_dcm_comments(text: str) -> str:
    # Remove /* ... */, and lines starting with //, *, ;
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)^\s*(//|\*|;).*$", "", text)
    return text

def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # Collapse exotic whitespace to simple spaces (keep \n and \t)
    text = re.sub(r"[^\S\n\t]+", " ", text)
    return text

# =========================================================
# Parser (2D data + 1D curves + scalars)
# =========================================================
_NUM_CLEAN = re.compile(r"[^0-9eE+\-.]")  # keep digits/signs/dot/exponent

def _to_float(tok: str) -> float:
    cleaned = _NUM_CLEAN.sub("", tok)
    if cleaned in ("", "+", "-", ".", "+.", "-."):
        raise ValueError(f"Bad numeric token: {tok!r}")
    return float(cleaned)

def _extract_blocks(text: str) -> list[str]:
    """Return every 'KENNFELD/GRUPPENKENNFELD/KENNLINIE/FESTWERT ... END' block."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if _HEADER_START_RE.match(line):
            buf = [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if lines[i].strip() == "END":
                    i += 1
                    break
                i += 1
            blocks.append("\n".join(buf))
        else:
            i += 1
    return blocks

def _parse_single_block(block_text: str):
    """
    Parse one block (2D, 1D, or scalar).
    Returns:
      - x : np.ndarray | None
      - y : Optional[np.ndarray] (None for 1D/scalars)
      - z : np.ndarray (2D for data; 1D for KENNLINIE; 0-D/1-elt for FESTWERT)
      - df: DataFrame
          * 2D: rows=Y, cols=X
          * 1D: index=X, single column 'W [unit]'
          * 0D (FESTWERT): single row, single column 'W [unit]'
      - meta: dict(kind, name, nx, ny, dim, units)
    """
    kind: Optional[str] = None
    name: Optional[str] = None
    nx: Optional[int] = None
    ny: Optional[int] = None
    units = {"X": None, "Y": None, "W": None}

    x_vals: list[float] = []
    y_vals: list[float] = []
    w_vals_1d: list[float] = []   # KENNLINIE (1D)
    w_scalar: list[float] = []    # FESTWERT (scalar)
    rows_2d: list[list[float]] = []

    lines = block_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        # ---- Headers ----
        m2 = _HEADER_PARSE_2D_RE.match(line)
        if m2:
            kind = m2.group(1).upper()           # 'KENNFELD' or 'GRUPPENKENNFELD'
            name = m2.group(2)
            nx = int(m2.group(3))
            ny = int(m2.group(4))
            continue

        m1 = _HEADER_PARSE_1D_RE.match(line)
        if m1:
            kind = m1.group(1).upper()           # 'KENNLINIE'
            name = m1.group(2)
            nx = int(m1.group(3))
            ny = 1
            continue

        ms = _HEADER_PARSE_SCALAR_RE.match(line)
        if ms:
            kind = ms.group(1).upper()           # 'FESTWERT'
            name = ms.group(2)
            nx = int(ms.group(3)) if ms.group(3) else 1  # default 1
            ny = 1
            continue

        # ---- Units ----
        if line.startswith("EINHEIT_"):
            m = re.match(r'EINHEIT_(X|Y|W)\s+"([^"]+)"', line)
            if m:
                units[m.group(1)] = m.group(2)
            continue

        # ---- X breakpoints ----
        if line.startswith("ST/X"):
            x_vals.extend(_to_float(t) for t in line.split()[1:])
            continue

        # ---- 2D rows: ST/Y + following WERT ----
        if line.startswith("ST/Y"):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad ST/Y line: {line}")
            yval = _to_float(parts[1])
            y_vals.append(yval)

            pending: list[float] = []
            while len(pending) < (nx or 0) and i < len(lines):
                nxt = lines[i].strip()
                if not nxt:
                    i += 1; continue
                if nxt.startswith("WERT"):
                    pending.extend(_to_float(tok) for tok in nxt.split()[1:])
                    i += 1
                else:
                    break
            if nx is not None and len(pending) != nx:
                raise ValueError(f"Row at Y={yval} has {len(pending)} values, expected {nx}.")
            rows_2d.append(pending)
            continue

        # ---- Values (KENNLINIE or FESTWERT) ----
        if line.startswith("WERT"):
            vals = [_to_float(tok) for tok in line.split()[1:]]
            if kind == "KENNLINIE":
                w_vals_1d.extend(vals)
            elif kind == "FESTWERT":
                w_scalar.extend(vals)
            else:
                # For safety: if header wasn't matched yet, store for later?
                # Here we ignore; KENNFELD/GRUPPENKENNFELD values are read with ST/Y loop.
                pass
            continue

        if line == "END":
            break

    # ---------------- Validations & DataFrames ----------------
    if kind is None or name is None or nx is None or ny is None:
        raise ValueError("Missing header with dimensions (KENNFELD/GRUPPENKENNFELD/KENNLINIE/FESTWERT).")

    # FESTWERT (scalar)
    if kind == "FESTWERT":
        if len(w_scalar) < 1:
            raise ValueError(f"[{name}] No WERT value found for FESTWERT.")
        if len(w_scalar) != 1:
            # If more than one value accidentally provided, take the first
            w_scalar = [w_scalar[0]]
        val = float(w_scalar[0])
        x = None
        y = None
        z = np.array(val)  # 0-D scalar array

        df = pd.DataFrame(
            {f"W [{units['W'] or ''}]": [val]},
            index=pd.Index(["value"], name="Scalar")
        )
        meta = {"kind": kind, "dim": 0, "name": name, "nx": 1, "ny": 1, "units": units}
        return x, y, z, df, meta

    # KENNLINIE (1D)
    if kind == "KENNLINIE":
        if len(x_vals) != nx:
            raise ValueError(f"[{name}] X count {len(x_vals)} != nx {nx}")
        if len(w_vals_1d) != nx:
            raise ValueError(f"[{name}] W count {len(w_vals_1d)} != nx {nx}")

        x = np.array(x_vals, dtype=float)
        y = None
        z = np.array(w_vals_1d, dtype=float)

        df = pd.DataFrame(
            {f"W [{units['W'] or ''}]": z},
            index=pd.Index(x, name=f"X [{units['X'] or ''}]")
        )
        meta = {"kind": kind, "dim": 1, "name": name, "nx": nx, "ny": 1, "units": units}
        return x, y, z, df, meta

    # 2D data (KENNFELD/GRUPPENKENNFELD)
    if len(x_vals) != nx:
        raise ValueError(f"[{name}] X breakpoint count {len(x_vals)} != nx {nx}.")
    if len(rows_2d) != ny:
        if len(rows_2d) == len(y_vals):
            ny = len(rows_2d)  # recover if header ny mismatched
        else:
            raise ValueError(f"[{name}] Parsed {len(rows_2d)} rows, expected {ny}.")

    z2d = np.array(rows_2d, dtype=float)  # (ny, nx)
    x2d = np.array(x_vals, dtype=float)
    y2d = np.array(y_vals, dtype=float)

    df2d = pd.DataFrame(z2d, index=y2d, columns=x2d)
    df2d.index.name = f"Y [{units['Y'] or ''}]"
    df2d.columns.name = f"X [{units['X'] or ''}]"

    meta2d = {"kind": kind, "dim": 2, "name": name, "nx": nx, "ny": ny, "units": units}
    return x2d, y2d, z2d, df2d, meta2d

# Public APIs
def parse_all_kennfelds(
    text: str,
    include_names: Iterable[str] | None = None,
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.DataFrame, dict]]:
    """
    Parse ALL blocks (2D, 1D, and scalar).
    Returns OrderedDict:
      name -> (x, y_or_None, z, df, meta)
    """
    name_filter = {n.strip() for n in include_names} if include_names else None
    out: "OrderedDict[str, Tuple[Any, ...]]" = OrderedDict()
    for block in _extract_blocks(text):
        try:
            x, y, z, df, meta = _parse_single_block(block)
        except Exception as e:
            print(f"[skip] Failed to parse a block: {e}")
            continue
        nm = meta["name"]
        if name_filter and nm not in name_filter:
            continue
        out[nm] = (x, y, z, df, meta)

    if name_filter and not out:
        raise ValueError(f"No blocks matched include_names={sorted(name_filter)}")
    if not out:
        raise ValueError("No KENN* blocks were found in the input.")
    return out

# =========================================================
# Excel (in-memory bytes)
# =========================================================
_INVALID_SHEET = re.compile(r'[:\\/?*\[\]]')

def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    base = _INVALID_SHEET.sub("_", name)[:31] or "Sheet"
    s = base; i = 2
    while s in used:
        suf = f"_{i}"; s = base[: 31 - len(suf)] + suf; i += 1
    used.add(s); return s

def build_excel_bytes(
    data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.DataFrame, dict]],
    add_summary: bool = True,
    round_digits: int | None = None,
) -> bytes:
    """
    Writes:
      - 2D data: df with Y index and X columns
      - 1D curves: df with X index and one 'W [unit]' column
      - 0D FESTWERT: single row 'value' x single column 'W [unit]'
    """
    buf = io.BytesIO()
    used = set()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if add_summary:
            summary = [{
                "Map": name,
                "Kind": meta["kind"],
                "Dim": meta["dim"],
                "Rows (ny)": meta["ny"],
                "Cols (nx)": meta["nx"],
                "Unit X": meta["units"]["X"],
                "Unit Y": meta["units"]["Y"],
                "Unit W": meta["units"]["W"],
            } for name, (_, _, _, df, meta) in data.items()]
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        for name, (_, _, _, df, meta) in data.items():
            sheet = _sanitize_sheet_name(name, used)
            dfw = df.round(round_digits) if round_digits is not None else df
            dfw.to_excel(writer, sheet_name=sheet)

            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 1)
            ws.set_column(0, 0, 14)  # index
            for j, col in enumerate(df.columns, start=1):
                ws.set_column(j, j, max(10, min(40, len(str(col)) + 2)))
    return buf.getvalue()

def load_parse_build_excel(
    source: str | os.PathLike | bytes,
    include_names: Iterable[str] | None = None,
    round_digits: int | None = None,
):
    text = load_kennfeld_text(source)
    data = parse_all_kennfelds(text, include_names=include_names)
    excel_bytes = build_excel_bytes(data, add_summary=True, round_digits=round_digits)
    return excel_bytes, data


# ---------- MATLAB .m writer ----------
import io as _io

def _matlab_ident(name: str) -> str:
    """Make a safe MATLAB identifier (for struct field names & vars)."""
    s = re.sub(r'[^A-Za-z0-9_]', '_', name.strip())
    if not s or not s[0].isalpha():
        s = 'm_' + s
    return s

def _unique_ident(suggest: str, used: set[str]) -> str:
    """Ensure the MATLAB identifier is unique within 'used'."""
    base = _matlab_ident(suggest)
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        cand = f"{base}_{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1

def _escq(s: str | None) -> str:
    """Escape single quotes for MATLAB strings."""
    return (s or "").replace("'", "''")

def _fmt_scalar(val: float, ndp: int | None = None) -> str:
    return f"{val:.{ndp}f}" if ndp is not None else f"{val:.10g}"

def _fmt_vec(arr: np.ndarray, ndp: int | None = None) -> str:
    """Format a 1D numpy array as a single-line MATLAB vector."""
    a = np.asarray(arr).reshape(-1)
    if a.size == 0:
        return "[]"
    def fmt(v): return f"{v:.{ndp}f}" if ndp is not None else f"{v:.10g}"
    return "[" + " ".join(fmt(v) for v in a.tolist()) + "]"

def _fmt_mat(mat: np.ndarray, ndp: int | None = None) -> str:
    """Format a 2D numpy array as a single-line MATLAB matrix."""
    m = np.asarray(mat)
    if m.size == 0:
        return "[]"
    def fmt(v): return f"{v:.{ndp}f}" if ndp is not None else f"{v:.10g}"
    rows = [" ".join(fmt(v) for v in row) for row in m.tolist()]
    return "[" + " ; ".join(rows) + "]"

# kennfeld_tools.py  (only the MATLAB writer parts shown)

def build_matlab_m_bytes(
    data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.DataFrame, dict]],
    round_digits: int | None = None,
    top_var: str = "data",
    alias_map: Optional[Dict[str, str]] = None,   # ⬅️ NEW
) -> bytes:
    """
    Build a single MATLAB .m script that defines a struct:
        data.<Map>.X / .Y / .Z (2D)
        data.<Map>.X / .W       (KENNLINIE, 1D)
        data.<Map>.W            (FESTWERT, scalar)
        data.<Map>.units, .kind, .nx, .ny, .name

    If alias_map is provided (orig_name -> alias), the struct field name and .name
    string are taken from the alias (fallback to original when missing/blank).
    """
    b = _io.StringIO()
    b.write("%% Auto-generated by Kennfeld Converter\n")
    #b.write(f"{top_var} = struct();\n\n")

    used_fields: set[str] = set()  # ensure unique MATLAB field names

    for orig_name, (x, y, z, df, meta) in data.items():
        # Resolve alias (for field + .name)
        alias = (alias_map or {}).get(orig_name, orig_name) or orig_name
        field = _matlab_ident(alias)

        # de-duplicate sanitized field names by adding suffixes
        base = field
        k = 2
        while field in used_fields:
            field = f"{base}_{k}"
            k += 1
        used_fields.add(field)

        kind  = meta.get("kind", "KENNFELD")
        dim   = int(meta.get("dim", 2))
        nx    = int(meta["nx"])
        ny    = int(meta["ny"])
        ux    = _escq(meta["units"].get("X"))
        uy    = _escq(meta["units"].get("Y"))
        uw    = _escq(meta["units"].get("W"))

        # Use alias in the comment header as well
        b.write(f"% ---- {alias} ({kind}, dim={dim}) ----\n")
        # b.write(f"{top_var}.{field} = struct();\n")
        # b.write(f"{top_var}.{field}.name = '{_escq(alias)}';\n")  # <-- alias stored in .name
        # b.write(f"{top_var}.{field}.kind = '{_escq(kind)}';\n")
        # b.write(f"{top_var}.{field}.nx = {nx};\n")
        # b.write(f"{top_var}.{field}.ny = {ny};\n")
        # b.write(f"{top_var}.{field}.units = struct('X','{ux}','Y','{uy}','W','{uw}');\n")

        if dim == 0:  # FESTWERT: scalar
            val = float(np.asarray(z).reshape(-1)[0])
            b.write(f"{field}_W = {_fmt_scalar(val, round_digits)};\n\n")
            continue

        b.write(f"{field}_X = {_fmt_vec(np.asarray(x), round_digits)};\n")
        if dim == 1:
            b.write(f"{field}_W = {_fmt_vec(np.asarray(z), round_digits)};\n\n")
        else:
            b.write(f"{field}_Y = {_fmt_vec(np.asarray(y), round_digits)};\n")
            b.write(f"{field}_Z = {_fmt_mat(np.asarray(z), round_digits)};\n\n")

    b.write(f"% Access example: {top_var}.<MapName>.W or .X/.Y/.Z\n")
    return b.getvalue().encode("utf-8")


def load_parse_build_m(
    source: str | os.PathLike | bytes,
    include_names: Iterable[str] | None = None,
    round_digits: int | None = None,
    top_var: str = "data",
    alias_map: Optional[Dict[str, str]] = None,  # ⬅️ NEW
) -> tuple[bytes, Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.DataFrame, dict]]]:
    """
    Convenience: load -> parse -> build MATLAB .m bytes.
    """
    text = load_kennfeld_text(source)
    data = parse_all_kennfelds(text, include_names=include_names)
    m_bytes = build_matlab_m_bytes(data, round_digits=round_digits, top_var=top_var, alias_map=alias_map)
    return m_bytes, data
