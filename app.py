# app.py
# ============================================================
# PNTOPSISForge (Conference UI) - Multi-tab Streamlit App
# Fixes included:
# 1) Refined summary wording (your preferred structure)
# 2) Sample dataset download format matches working Excel (download -> upload works)
# 3) Cropped UI callout fixed (card with overflow visible)
# 4) Larger fonts for numbers/tables/data editor (CSS + heights)
# ============================================================

import io
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# PDF report
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# -----------------------------
# PNS Linguistic Tables (numeric triplets)
# -----------------------------
PNS_TABLES: Dict[int, Dict[int, Tuple[float, float, float]]] = {
    5: {
        1: (0.10, 0.85, 0.90),
        2: (0.30, 0.65, 0.70),
        3: (0.50, 0.45, 0.45),
        4: (0.70, 0.25, 0.20),
        5: (0.90, 0.10, 0.05),
    },
    7: {
        1: (0.10, 0.80, 0.90),
        2: (0.20, 0.70, 0.80),
        3: (0.35, 0.60, 0.60),
        4: (0.50, 0.40, 0.45),
        5: (0.65, 0.30, 0.25),
        6: (0.80, 0.20, 0.15),
        7: (0.90, 0.10, 0.10),
    },
    9: {
        1: (0.05, 0.90, 0.95),
        2: (0.10, 0.85, 0.90),
        3: (0.20, 0.80, 0.75),
        4: (0.35, 0.65, 0.60),
        5: (0.50, 0.50, 0.45),
        6: (0.65, 0.35, 0.30),
        7: (0.80, 0.25, 0.20),
        8: (0.90, 0.15, 0.10),
        9: (0.95, 0.05, 0.05),
    },
    11: {
        1: (0.05, 0.90, 0.95),
        2: (0.10, 0.80, 0.85),
        3: (0.20, 0.70, 0.75),
        4: (0.30, 0.60, 0.65),
        5: (0.40, 0.50, 0.55),
        6: (0.50, 0.45, 0.45),
        7: (0.60, 0.40, 0.35),
        8: (0.70, 0.30, 0.25),
        9: (0.80, 0.20, 0.15),
        10: (0.90, 0.15, 0.10),
        11: (0.95, 0.05, 0.05),
    },
}


# -----------------------------
# Helpers
# -----------------------------
def is_bc_row(values: List[str]) -> bool:
    cleaned = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        cleaned.append(s.upper())
    if not cleaned:
        return False
    return all(x in {"B", "C"} for x in cleaned)


def read_uploaded_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def coerce_int_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        try:
            out[c] = out[c].apply(lambda x: int(str(x).strip()))
        except Exception as e:
            raise ValueError(f"Column '{c}' contains non-integer values. ({e})")
    return out


def validate_score_range(df_int: pd.DataFrame, scale: int) -> None:
    lo, hi = 1, scale
    bad = (df_int < lo) | (df_int > hi)
    if bad.values.any():
        idx = np.argwhere(bad.values)[0]
        r, c = idx[0], idx[1]
        raise ValueError(
            f"Invalid crisp score: {df_int.iloc[r, c]}. Allowed range for {scale}-point scale is {lo}..{hi}."
        )


def map_crisp_to_pns(df_int: pd.DataFrame, scale: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = PNS_TABLES[scale]
    m, n = df_int.shape
    tau = np.zeros((m, n), dtype=float)
    xi = np.zeros((m, n), dtype=float)
    eta = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            t, x, e = table[int(df_int.iat[i, j])]
            tau[i, j], xi[i, j], eta[i, j] = t, x, e
    return tau, xi, eta


def normalize_pns(
    tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, crit_types: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, n = tau.shape
    tau_n = np.zeros_like(tau)
    xi_n = np.zeros_like(xi)
    eta_n = np.zeros_like(eta)

    for j in range(n):
        ctype = crit_types[j].upper()
        if ctype == "B":
            tmax, xmax, emax = float(np.max(tau[:, j])), float(np.max(xi[:, j])), float(np.max(eta[:, j]))
            if tmax == 0 or xmax == 0 or emax == 0:
                raise ValueError(f"Normalization failed: max component is 0 for criterion {j+1}.")
            tau_n[:, j] = tau[:, j] / tmax
            xi_n[:, j] = xi[:, j] / xmax
            eta_n[:, j] = eta[:, j] / emax
        elif ctype == "C":
            tmin, xmin, emin = float(np.min(tau[:, j])), float(np.min(xi[:, j])), float(np.min(eta[:, j]))
            if np.any(tau[:, j] == 0) or np.any(xi[:, j] == 0) or np.any(eta[:, j] == 0):
                raise ValueError(f"Normalization failed: a component is 0 in criterion {j+1}.")
            tau_n[:, j] = tmin / tau[:, j]
            xi_n[:, j] = xmin / xi[:, j]
            eta_n[:, j] = emin / eta[:, j]
        else:
            raise ValueError("Criterion types must be only 'B' or 'C'.")
    return tau_n, xi_n, eta_n


def apply_weights(
    tau_n: np.ndarray, xi_n: np.ndarray, eta_n: np.ndarray, w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tau_n * w.reshape(1, -1), xi_n * w.reshape(1, -1), eta_n * w.reshape(1, -1)


def compute_ideals(tau_w: np.ndarray, xi_w: np.ndarray, eta_w: np.ndarray, crit_types: List[str]):
    n = tau_w.shape[1]
    tau_p = np.zeros(n)
    xi_p = np.zeros(n)
    eta_p = np.zeros(n)
    tau_n = np.zeros(n)
    xi_n = np.zeros(n)
    eta_n = np.zeros(n)

    for j in range(n):
        if crit_types[j].upper() == "B":
            tau_p[j], xi_p[j], eta_p[j] = np.max(tau_w[:, j]), np.min(xi_w[:, j]), np.min(eta_w[:, j])
            tau_n[j], xi_n[j], eta_n[j] = np.min(tau_w[:, j]), np.max(xi_w[:, j]), np.max(eta_w[:, j])
        else:
            tau_p[j], xi_p[j], eta_p[j] = np.min(tau_w[:, j]), np.max(xi_w[:, j]), np.max(eta_w[:, j])
            tau_n[j], xi_n[j], eta_n[j] = np.max(tau_w[:, j]), np.min(xi_w[:, j]), np.min(eta_w[:, j])

    return tau_p, xi_p, eta_p, tau_n, xi_n, eta_n


# -----------------------------
# Distance measures
# -----------------------------
def dist_pn_euclidean(tau_row, xi_row, eta_row, tau_ideal, xi_ideal, eta_ideal) -> float:
    n = tau_row.shape[0]
    diff2 = (tau_row - tau_ideal) ** 2 + (xi_row - xi_ideal) ** 2 + (eta_row - eta_ideal) ** 2
    return float(math.sqrt((1.0 / (3.0 * n)) * float(np.sum(diff2))))


def dist_pn_hamming(tau_row, xi_row, eta_row, tau_ideal, xi_ideal, eta_ideal) -> float:
    n = tau_row.shape[0]
    s = np.abs(tau_row - tau_ideal) + np.abs(xi_row - xi_ideal) + np.abs(eta_row - eta_ideal)
    return float((1.0 / (3.0 * n)) * float(np.sum(s)))


def dist_fiq(tau_row, xi_row, eta_row, tau_ideal, xi_ideal, eta_ideal) -> float:
    """
    FIQ distance per your formula implementation (automated parameters).
    """
    n = tau_row.shape[0]

    d_tau = np.abs(tau_row - tau_ideal)
    d_xi = np.abs(xi_row - xi_ideal)
    d_eta = np.abs(eta_row - eta_ideal)

    w_tau = 1.0 - (xi_row * xi_ideal)
    w_eta = 1.0 - (xi_row * xi_ideal)
    w_xi = 1.0 + (np.abs(tau_row - eta_ideal) + np.abs(eta_row - tau_ideal)) / 2.0

    p_tau = 1.0 + (xi_row + xi_ideal) / 2.0
    p_eta = 1.0 + (xi_row + xi_ideal) / 2.0
    p_xi = 2.0 - np.abs(tau_row - eta_ideal)

    p = 2.0 - (xi_row * xi_ideal)
    p = np.maximum(p, 1e-9)

    inner = (w_tau * (d_tau ** p_tau)) + (w_xi * (d_xi ** p_xi)) + (w_eta * (d_eta ** p_eta))
    contrib = inner ** (1.0 / p)

    return float((1.0 / (3.0 * n)) * float(np.sum(contrib)))


DIST_FUNCS = {
    "PN-Euclidean": dist_pn_euclidean,
    "PN-Hamming": dist_pn_hamming,
    "FIQ": dist_fiq,
}


def format_triplets(tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, decimals: int = 2) -> pd.DataFrame:
    m, n = tau.shape
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = f"({tau[i,j]:.{decimals}f}, {xi[i,j]:.{decimals}f}, {eta[i,j]:.{decimals}f})"
    return pd.DataFrame(out)


def compute_result_for_distance(
    dist_name: str,
    tau_w: np.ndarray,
    xi_w: np.ndarray,
    eta_w: np.ndarray,
    tau_p: np.ndarray,
    xi_p: np.ndarray,
    eta_p: np.ndarray,
    tau_n: np.ndarray,
    xi_n: np.ndarray,
    eta_n: np.ndarray,
    alt_names: List[str],
) -> pd.DataFrame:
    dist_fn = DIST_FUNCS[dist_name]
    m_alt = tau_w.shape[0]
    S_plus = np.zeros(m_alt, dtype=float)
    S_minus = np.zeros(m_alt, dtype=float)

    for i in range(m_alt):
        S_plus[i] = dist_fn(tau_w[i, :], xi_w[i, :], eta_w[i, :], tau_p, xi_p, eta_p)
        S_minus[i] = dist_fn(tau_w[i, :], xi_w[i, :], eta_w[i, :], tau_n, xi_n, eta_n)

    Pi = S_minus / (S_plus + S_minus)
    result = pd.DataFrame({"S_i_plus": S_plus, "S_i_minus": S_minus, "P_i": Pi}, index=alt_names)
    result["Rank"] = (-result["P_i"]).rank(method="dense").astype(int)
    return result.sort_values(["Rank", "P_i"], ascending=[True, False])


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    return buf.read()


# ✅ FIXED sample generator: matches the working format
def sample_dataset_bytes_excel(scale: int) -> bytes:
    """
    Output format:
    - Header row: Alt, C1..C5
    - Row 2: criterion types (B/C) under C1..C5
    - Rows 3+: alternatives with integer crisp scores
    """
    crit_names = ["C1", "C2", "C3", "C4", "C5"]
    types_row = [""] + ["B", "C", "B", "B", "C"]

    hi = int(scale)

    def clip(v: int) -> int:
        return max(1, min(hi, int(v)))

    data = [
        ["A1", clip(7), clip(5), clip(8), clip(6), clip(4)],
        ["A2", clip(6), clip(3), clip(7), clip(5), clip(6)],
        ["A3", clip(8), clip(4), clip(6), clip(7), clip(5)],
        ["A4", clip(5), clip(6), clip(5), clip(4), clip(7)],
        ["A5", clip(9), clip(2), clip(8), clip(6), clip(3)],
    ]

    df_out = pd.DataFrame(data, columns=["Alt"] + crit_names)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Input_Matrix", index=False)
        pd.DataFrame([types_row], columns=["Alt"] + crit_names).to_excel(
            writer, sheet_name="Input_Matrix", index=False, header=False, startrow=1
        )

    buf.seek(0)
    return buf.read()


def build_pdf_report(
    distance_name: str,
    scale: int,
    crit_meta: pd.DataFrame,
    result: pd.DataFrame,
    interpretation: str,
    elapsed_sec: float,
    top_k: int = 10,
) -> bytes:
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    heading = styles["Heading2"]
    h1 = styles["Heading1"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
    )

    story = []
    story.append(Paragraph("PNTOPSISForge Report", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    story.append(Paragraph(f"Computation time (primary ranking): {elapsed_sec:.6f} seconds", normal))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Settings", heading))
    story.append(Paragraph(f"Linguistic scale: {scale}-point", normal))
    story.append(Paragraph(f"Primary distance: {distance_name}", normal))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("Criteria metadata", heading))
    meta_tbl_data = [list(crit_meta.columns)] + crit_meta.values.tolist()
    meta_tbl = Table(meta_tbl_data, hAlign="LEFT")
    meta_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Ranking results", heading))
    res = result.copy().reset_index().rename(columns={"index": "Alternative"})
    res = res.sort_values(["Rank", "P_i"], ascending=[True, False]).head(top_k)
    res_tbl_data = [list(res.columns)] + res.round(6).values.tolist()
    res_tbl = Table(res_tbl_data, hAlign="LEFT")
    res_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(res_tbl)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Interpretation", heading))
    story.append(Paragraph(interpretation, normal))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(page_title="PNTOPSISForge", layout="wide")

# --- Blue theme + larger fonts (tables and editors too)
st.markdown(
    """
<style>
/* Global typography */
html, body, [class*="css"] { font-size: 18px !important; }

/* Headings */
h1 { color: #0B5394; font-size: 40px !important; }
h2 { color: #0B5394; font-size: 28px !important; }
h3 { color: #1F77B4; font-size: 22px !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #F2F6FC; }

/* Tables + Dataframes */
table { font-size: 18px !important; }
thead tr th, tbody tr td { font-size: 18px !important; }
[data-testid="stDataFrame"] * { font-size: 18px !important; }
[data-testid="stDataEditor"] * { font-size: 18px !important; }

/* Data editor inputs */
[data-testid="stDataEditor"] input,
[data-testid="stDataEditor"] textarea,
[data-testid="stDataEditor"] [contenteditable="true"] {
  font-size: 18px !important;
  line-height: 1.35 !important;
}

/* Buttons */
button[kind="primary"] { background-color: #1F77B4 !important; color: white !important; font-size: 18px !important; }
button[kind="secondary"] { font-size: 18px !important; }

/* Layout spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Callout card (fix cropped / overflow issues) */
.pnt-card{
  border: 1px solid rgba(31, 119, 180, 0.25);
  background: #F7FBFF;
  padding: 14px 16px;
  border-radius: 14px;
  overflow: visible !important;
  line-height: 1.55;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("PNTOPSISForge")
st.caption("A Pythagorean Neutrosophic TOPSIS Decision Support System (Conference Demo UI)")

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")

scale = st.sidebar.selectbox("Linguistic scale", options=[5, 7, 9, 11], index=2)  # default 9-point
distance_name = st.sidebar.selectbox("Primary distance", options=["FIQ", "PN-Euclidean", "PN-Hamming"], index=0)
sensitivity_on = st.sidebar.checkbox("Enable sensitivity analysis (compare all distances)", value=True)
decimals = st.sidebar.slider("Triplet display decimals", 2, 6, 2)

st.sidebar.subheader("Try the app quickly")
st.sidebar.download_button(
    "Download sample Excel (works with re-upload)",
    data=sample_dataset_bytes_excel(scale=scale),
    file_name=f"pntopsisforge_sample_{scale}scale.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

weight_mode = st.sidebar.radio("Criteria weights", ["Equal Weights", "Manual Weights"], index=0)

# Linguistic table for reference
table_df = pd.DataFrame(
    [{"Score": k, "tau": v[0], "xi": v[1], "eta": v[2]} for k, v in PNS_TABLES[scale].items()]
).sort_values("Score")


# -----------------------------
# Tabs layout (demo-ready)
# -----------------------------
tab_input, tab_run, tab_results, tab_export = st.tabs(
    ["1) Input", "2) Compute", "3) Results", "4) Export"]
)

# Store computed outputs in session_state so tabs behave nicely
if "computed" not in st.session_state:
    st.session_state.computed = False
if "bundle" not in st.session_state:
    st.session_state.bundle = {}


# ============================================================
# TAB 1: INPUT
# ============================================================
with tab_input:
    st.markdown(
        """<div class="pnt-card">
        <b>How to start</b><br/>
        1) Download the sample Excel (left sidebar) and upload it back, or upload your own file.<br/>
        2) Choose the linguistic scale, distance measure, and weights.<br/>
        3) If benefit/cost row is missing, set it here before computing.<br/>
        </div>""",
        unsafe_allow_html=True,
    )

    st.subheader("Upload preview")

    raw_df: Optional[pd.DataFrame] = None
    if uploaded is not None:
        try:
            raw_df = read_uploaded_file(uploaded)
            st.success(f"Loaded file: {uploaded.name}")
            st.dataframe(raw_df, use_container_width=True, height=320)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
    else:
        st.info("No file uploaded yet. You can still test using the sample Excel from the sidebar.")
        st.stop()

    # Determine alternative names column
    df = raw_df.copy()
    first_col_values = df.iloc[:, 0].tolist()
    first_col_is_alt = True
    try:
        _ = [int(str(v).strip()) for v in first_col_values[: min(10, len(first_col_values))]]
        first_col_is_alt = False
    except Exception:
        first_col_is_alt = True

    if first_col_is_alt:
        alt_names_all = df.iloc[:, 0].astype(str).tolist()
        df_mat = df.iloc[:, 1:].copy()
        crit_names = [str(c) for c in df_mat.columns]
    else:
        alt_names_all = [f"A{i+1}" for i in range(df.shape[0])]
        df_mat = df.copy()
        crit_names = [str(c) for c in df_mat.columns]

    # Detect B/C row in first row of df_mat
    first_row = df_mat.iloc[0, :].tolist()
    has_bc = is_bc_row(first_row)

    if has_bc:
        detected_types = [str(x).strip().upper() for x in first_row]
        df_scores = df_mat.iloc[1:, :].copy()
        alt_names = alt_names_all[1:]
        crit_types = detected_types
        st.info("Detected a criterion type row (B/C) at the top of the matrix.")
    else:
        df_scores = df_mat.copy()
        alt_names = alt_names_all
        crit_types = ["B"] * len(crit_names)
        st.warning("No criterion type row detected. Defaulting all criteria to Benefit (B). Please adjust below.")

    st.subheader("Criterion types (Benefit/Cost)")
    type_df = pd.DataFrame([crit_types], columns=crit_names, index=["Type (B/C)"])
    edited_type_df = st.data_editor(type_df, use_container_width=True, height=110, key="crit_types_editor")
    crit_types = [str(edited_type_df.iloc[0, j]).strip().upper() for j in range(len(crit_names))]

    if any(t not in {"B", "C"} for t in crit_types):
        st.error("Criterion types must be only 'B' or 'C' for every criterion.")
        st.stop()

    st.subheader("Crisp decision matrix (must be integers)")
    # Clean score matrix
    crisp_df = df_scores.copy()
    crisp_df = crisp_df.loc[:, ~crisp_df.columns.astype(str).str.contains("^Unnamed", regex=True)]
    crisp_df.columns = crit_names[: crisp_df.shape[1]]
    crisp_df = crisp_df.reset_index(drop=True)

    try:
        crisp_int = coerce_int_matrix(crisp_df)
        validate_score_range(crisp_int, scale)
    except Exception as e:
        st.error(f"Input validation error: {e}")
        st.stop()

    crisp_show = crisp_int.copy()
    crisp_show.index = alt_names
    st.dataframe(crisp_show, use_container_width=True, height=320)

    st.subheader("Reference: Selected PNS linguistic table")
    st.dataframe(table_df, use_container_width=True, height=260)

    # Prepare weights
    n_criteria = len(crit_names)
    if weight_mode == "Equal Weights":
        w = np.array([1.0 / n_criteria] * n_criteria, dtype=float)
        st.info(f"Equal weights applied: each w_j = 1/{n_criteria} = {1.0/n_criteria:.6f}")
        st.dataframe(pd.DataFrame([w], columns=crit_names, index=["w"]), use_container_width=True, height=90)
    else:
        st.info("Manual weights: ensure the total sum is 1.")
        w_default = pd.DataFrame([[round(1.0 / n_criteria, 6)] * n_criteria], columns=crit_names, index=["w"])
        w_edit = st.data_editor(w_default, use_container_width=True, height=110, key="weights_editor")
        try:
            w = np.array([float(w_edit.iloc[0, j]) for j in range(n_criteria)], dtype=float)
        except Exception as e:
            st.error(f"Manual weights must be numeric. ({e})")
            st.stop()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum):
            st.error("Manual weights contain invalid numbers.")
            st.stop()
        if abs(w_sum - 1.0) > 1e-3:
            st.warning(f"Sum of weights = {w_sum:.6f} (should be 1.000000). Computation will proceed anyway.")

    # Save prepared inputs for the next tab
    st.session_state.bundle = {
        "crit_names": crit_names,
        "alt_names": alt_names,
        "crit_types": crit_types,
        "crisp_int": crisp_int,
        "crisp_show": crisp_show,
        "w": w,
    }
    st.session_state.computed = False


# ============================================================
# TAB 2: COMPUTE
# ============================================================
with tab_run:
    st.subheader("Run PNTOPSISForge")

    if not st.session_state.bundle:
        st.info("Go to the Input tab first and upload a valid dataset.")
        st.stop()

    st.markdown(
        """<div class="pnt-card">
        Click <b>Run PNTOPSISForge</b> to execute the full PNTOPSIS procedure with your selected settings.
        </div>""",
        unsafe_allow_html=True,
    )

    run = st.button("Run PNTOPSISForge", type="primary")

    if run:
        crit_names = st.session_state.bundle["crit_names"]
        alt_names = st.session_state.bundle["alt_names"]
        crit_types = st.session_state.bundle["crit_types"]
        crisp_int = st.session_state.bundle["crisp_int"]
        crisp_show = st.session_state.bundle["crisp_show"]
        w = st.session_state.bundle["w"]

        # compute
        t0 = time.perf_counter()

        tau, xi, eta = map_crisp_to_pns(crisp_int, scale)
        tau_n, xi_n, eta_n = normalize_pns(tau, xi, eta, crit_types)
        tau_w, xi_w, eta_w = apply_weights(tau_n, xi_n, eta_n, w)
        tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg = compute_ideals(tau_w, xi_w, eta_w, crit_types)

        t1 = time.perf_counter()
        primary_result = compute_result_for_distance(
            distance_name, tau_w, xi_w, eta_w, tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg, alt_names
        )
        t2 = time.perf_counter()

        pipeline_sec = t1 - t0
        primary_sec = t2 - t1

        # Build display dfs
        pns_df = format_triplets(tau, xi, eta, decimals=decimals)
        pns_df.columns = crit_names
        pns_df.index = alt_names

        norm_df = format_triplets(tau_n, xi_n, eta_n, decimals=decimals)
        norm_df.columns = crit_names
        norm_df.index = alt_names

        w_df2 = format_triplets(tau_w, xi_w, eta_w, decimals=decimals)
        w_df2.columns = crit_names
        w_df2.index = alt_names

        pis = pd.DataFrame({"tau+": tau_p, "xi+": xi_p, "eta+": eta_p}, index=crit_names)
        nis = pd.DataFrame({"tau-": tau_neg, "xi-": xi_neg, "eta-": eta_neg}, index=crit_names)
        pisnis = pd.concat([pis, nis], axis=1)

        # Summary (refined, your preferred structure)
        best_alt = primary_result.index[0]
        best_pi = float(primary_result.iloc[0]["P_i"])
        best_s_plus = float(primary_result.iloc[0]["S_i_plus"])
        best_s_minus = float(primary_result.iloc[0]["S_i_minus"])

        interpretation = (
            f"{best_alt} is the best alternative because it has the highest relative closeness "
            f"P_i = {best_pi:.6f}. Under {distance_name}, it achieves a smaller distance to the positive ideal "
            f"(S_i_plus = {best_s_plus:.6f}) and a larger distance from the negative ideal "
            f"(S_i_minus = {best_s_minus:.6f})."
        )

        # Sensitivity (optional)
        compare = None
        pi_compare = None
        sensitivity_sec = None

        if sensitivity_on:
            t3 = time.perf_counter()
            frames = {}
            for dname in ["PN-Euclidean", "PN-Hamming", "FIQ"]:
                frames[dname] = compute_result_for_distance(
                    dname, tau_w, xi_w, eta_w, tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg, alt_names
                )[["P_i", "Rank"]].rename(columns={"P_i": f"P_i ({dname})", "Rank": f"Rank ({dname})"})

            compare = pd.concat([frames["PN-Euclidean"], frames["PN-Hamming"], frames["FIQ"]], axis=1)
            compare = compare.loc[primary_result.index]  # keep primary order

            pi_compare = pd.DataFrame(
                {
                    "PN-Euclidean": frames["PN-Euclidean"][f"P_i (PN-Euclidean)"],
                    "PN-Hamming": frames["PN-Hamming"][f"P_i (PN-Hamming)"],
                    "FIQ": frames["FIQ"][f"P_i (FIQ)"],
                },
                index=primary_result.index,
            )
            t4 = time.perf_counter()
            sensitivity_sec = t4 - t3

        st.session_state.bundle.update(
            {
                "tau": tau,
                "xi": xi,
                "eta": eta,
                "tau_n": tau_n,
                "xi_n": xi_n,
                "eta_n": eta_n,
                "tau_w": tau_w,
                "xi_w": xi_w,
                "eta_w": eta_w,
                "tau_p": tau_p,
                "xi_p": xi_p,
                "eta_p": eta_p,
                "tau_neg": tau_neg,
                "xi_neg": xi_neg,
                "eta_neg": eta_neg,
                "pns_df": pns_df,
                "norm_df": norm_df,
                "w_df2": w_df2,
                "pisnis": pisnis,
                "primary_result": primary_result,
                "interpretation": interpretation,
                "pipeline_sec": pipeline_sec,
                "primary_sec": primary_sec,
                "compare": compare,
                "pi_compare": pi_compare,
                "sensitivity_sec": sensitivity_sec,
            }
        )
        st.session_state.computed = True

        st.success("Computation completed. Go to the Results tab to view outputs.")
        st.info(f"Time - preprocessing: {pipeline_sec:.6f} s | primary ranking: {primary_sec:.6f} s")


# ============================================================
# TAB 3: RESULTS
# ============================================================
with tab_results:
    st.subheader("Results")

    if not st.session_state.computed:
        st.info("Run the computation first in the Compute tab.")
        st.stop()

    b = st.session_state.bundle
    crit_names = b["crit_names"]
    alt_names = b["alt_names"]

    st.markdown(
        f"""<div class="pnt-card">
        <b>Summary</b><br/>{b["interpretation"]}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Ranking table")
    st.dataframe(b["primary_result"], use_container_width=True, height=260)

    st.markdown("### Closeness chart (P_i)")
    pi_chart_df = b["primary_result"].reset_index().rename(columns={"index": "Alternative"})
    pi_chart_df["Alternative"] = pi_chart_df["Alternative"].astype(str)

    chart = (
        alt.Chart(pi_chart_df)
        .mark_bar(color="#1F77B4")
        .encode(
            x=alt.X("Alternative:N", sort=None, title="Alternatives"),
            y=alt.Y("P_i:Q", title="Closeness (P_i)"),
            tooltip=["Alternative:N", alt.Tooltip("P_i:Q", format=".4f")],
        )
        .properties(height=360)
    )
    text = (
        alt.Chart(pi_chart_df)
        .mark_text(dy=-10, size=16, color="#0B5394")
        .encode(x="Alternative:N", y="P_i:Q", text=alt.Text("P_i:Q", format=".4f"))
    )
    st.altair_chart((chart + text).configure_axis(labelFontSize=16, titleFontSize=18), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Converted PNS matrix (numeric triplets)")
        st.dataframe(b["pns_df"], use_container_width=True, height=320)
    with col2:
        st.markdown("### Normalized PNS matrix (numeric triplets)")
        st.dataframe(b["norm_df"], use_container_width=True, height=320)

    st.markdown("### Weighted normalized PNS matrix (numeric triplets)")
    st.dataframe(b["w_df2"], use_container_width=True, height=320)

    st.markdown("### PIS (V+) and NIS (V-) per criterion")
    st.dataframe(b["pisnis"], use_container_width=True, height=260)

    if sensitivity_on and b.get("compare") is not None:
        st.markdown("---")
        st.markdown("## Sensitivity analysis (side-by-side)")

        st.dataframe(b["compare"], use_container_width=True, height=280)

        pi_compare = b["pi_compare"]
        pi_compare_plot = pi_compare.reset_index().rename(columns={"index": "Alternative"})
        pi_compare_long = pi_compare_plot.melt(id_vars=["Alternative"], var_name="Distance", value_name="P_i")

        color_scale = alt.Scale(domain=["PN-Euclidean", "PN-Hamming", "FIQ"], range=["#1F77B4", "#0B5394", "#6FA8DC"])

        chart2 = (
            alt.Chart(pi_compare_long)
            .mark_bar()
            .encode(
                x=alt.X("Alternative:N", title="Alternatives"),
                y=alt.Y("P_i:Q", title="Closeness (P_i)"),
                color=alt.Color("Distance:N", scale=color_scale, title="Distance"),
                tooltip=["Alternative:N", "Distance:N", alt.Tooltip("P_i:Q", format=".4f")],
            )
            .properties(height=380)
        )

        st.altair_chart(
            chart2.configure_axis(labelFontSize=16, titleFontSize=18).configure_legend(labelFontSize=14, titleFontSize=14),
            use_container_width=True,
        )

        if b.get("sensitivity_sec") is not None:
            st.info(f"Sensitivity analysis time: {b['sensitivity_sec']:.6f} s")


# ============================================================
# TAB 4: EXPORT
# ============================================================
with tab_export:
    st.subheader("Export (Excel + PDF)")

    if not st.session_state.computed:
        st.info("Run the computation first in the Compute tab.")
        st.stop()

    b = st.session_state.bundle
    crit_names = b["crit_names"]
    crit_types = b["crit_types"]
    w = b["w"]

    meta = pd.DataFrame({"Criterion": crit_names, "Type (B/C)": crit_types, "Weight": w})

    # Excel
    sheets = {
        "Crisp_Matrix": b["crisp_show"],
        "PNS_Matrix": b["pns_df"],
        "Normalized": b["norm_df"],
        "Weighted_Normalized": b["w_df2"],
        "PIS_NIS": b["pisnis"],
        "Results_Primary": b["primary_result"],
        "Meta": meta.set_index("Criterion"),
        "Linguistic_Table": table_df.set_index("Score"),
    }

    xlsx_bytes = to_excel_bytes(sheets)
    st.download_button(
        "Download results (Excel)",
        data=xlsx_bytes,
        file_name="pntopsisforge_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # PDF
    pdf_bytes = build_pdf_report(
        distance_name=distance_name,
        scale=scale,
        crit_meta=meta,
        result=b["primary_result"],
        interpretation=b["interpretation"],
        elapsed_sec=b["primary_sec"],
        top_k=10,
    )

    st.download_button(
        "Download report (PDF)",
        data=pdf_bytes,
        file_name="pntopsisforge_report.pdf",
        mime="application/pdf",
    )