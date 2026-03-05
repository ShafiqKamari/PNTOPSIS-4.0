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


# ============================================================
# PNTOPSISForge v1.0 (Conference UI)
# ============================================================

PNS_TABLES: Dict[int, Dict[int, Tuple[float, float, float]]] = {
    5: {1: (0.10, 0.85, 0.90), 2: (0.30, 0.65, 0.70), 3: (0.50, 0.45, 0.45), 4: (0.70, 0.25, 0.20), 5: (0.90, 0.10, 0.05)},
    7: {1: (0.10, 0.80, 0.90), 2: (0.20, 0.70, 0.80), 3: (0.35, 0.60, 0.60), 4: (0.50, 0.40, 0.45), 5: (0.65, 0.30, 0.25), 6: (0.80, 0.20, 0.15), 7: (0.90, 0.10, 0.10)},
    9: {1: (0.05, 0.90, 0.95), 2: (0.10, 0.85, 0.90), 3: (0.20, 0.80, 0.75), 4: (0.35, 0.65, 0.60), 5: (0.50, 0.50, 0.45), 6: (0.65, 0.35, 0.30), 7: (0.80, 0.25, 0.20), 8: (0.90, 0.15, 0.10), 9: (0.95, 0.05, 0.05)},
    11: {1: (0.05, 0.90, 0.95), 2: (0.10, 0.80, 0.85), 3: (0.20, 0.70, 0.75), 4: (0.30, 0.60, 0.65), 5: (0.40, 0.50, 0.55), 6: (0.50, 0.45, 0.45), 7: (0.60, 0.40, 0.35), 8: (0.70, 0.30, 0.25), 9: (0.80, 0.20, 0.15), 10: (0.90, 0.15, 0.10), 11: (0.95, 0.05, 0.05)},
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
        if s:
            cleaned.append(s.upper())
    return bool(cleaned) and all(x in {"B", "C"} for x in cleaned)


def coerce_int_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].apply(lambda x: int(str(x).strip()))
    return out


def validate_score_range(df_int: pd.DataFrame, scale: int) -> None:
    lo, hi = 1, scale
    bad = (df_int < lo) | (df_int > hi)
    if bad.values.any():
        r, c = np.argwhere(bad.values)[0]
        raise ValueError(f"Invalid crisp score: {df_int.iloc[r, c]}. Allowed range for {scale}-point scale is {lo}..{hi}.")


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


def normalize_pns(tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, crit_types: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, n = tau.shape
    tau_n = np.zeros_like(tau)
    xi_n = np.zeros_like(xi)
    eta_n = np.zeros_like(eta)
    for j in range(n):
        ctype = crit_types[j].upper()
        if ctype == "B":
            tau_n[:, j] = tau[:, j] / float(np.max(tau[:, j]))
            xi_n[:, j] = xi[:, j] / float(np.max(xi[:, j]))
            eta_n[:, j] = eta[:, j] / float(np.max(eta[:, j]))
        elif ctype == "C":
            tau_n[:, j] = float(np.min(tau[:, j])) / tau[:, j]
            xi_n[:, j] = float(np.min(xi[:, j])) / xi[:, j]
            eta_n[:, j] = float(np.min(eta[:, j])) / eta[:, j]
        else:
            raise ValueError("Criterion types must be only 'B' or 'C'.")
    return tau_n, xi_n, eta_n


def apply_weights(tau_n: np.ndarray, xi_n: np.ndarray, eta_n: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w2 = w.reshape(1, -1)
    return tau_n * w2, xi_n * w2, eta_n * w2


def compute_ideals(tau_w: np.ndarray, xi_w: np.ndarray, eta_w: np.ndarray, crit_types: List[str]):
    n = tau_w.shape[1]
    tau_p = np.zeros(n); xi_p = np.zeros(n); eta_p = np.zeros(n)
    tau_n = np.zeros(n); xi_n = np.zeros(n); eta_n = np.zeros(n)
    for j in range(n):
        if crit_types[j].upper() == "B":
            tau_p[j], xi_p[j], eta_p[j] = np.max(tau_w[:, j]), np.min(xi_w[:, j]), np.min(eta_w[:, j])
            tau_n[j], xi_n[j], eta_n[j] = np.min(tau_w[:, j]), np.max(xi_w[:, j]), np.max(eta_w[:, j])
        else:
            tau_p[j], xi_p[j], eta_p[j] = np.min(tau_w[:, j]), np.max(xi_w[:, j]), np.max(eta_w[:, j])
            tau_n[j], xi_n[j], eta_n[j] = np.max(tau_w[:, j]), np.min(xi_w[:, j]), np.min(eta_w[:, j])
    return tau_p, xi_p, eta_p, tau_n, xi_n, eta_n


# -----------------------------
# Distances
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


DIST_FUNCS = {"PN-Euclidean": dist_pn_euclidean, "PN-Hamming": dist_pn_hamming, "FIQ": dist_fiq}


def compute_result_for_distance(
    dist_name: str,
    tau_w: np.ndarray, xi_w: np.ndarray, eta_w: np.ndarray,
    tau_p: np.ndarray, xi_p: np.ndarray, eta_p: np.ndarray,
    tau_n: np.ndarray, xi_n: np.ndarray, eta_n: np.ndarray,
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


def format_triplets(tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, decimals: int = 2) -> pd.DataFrame:
    m, n = tau.shape
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = f"({tau[i,j]:.{decimals}f}, {xi[i,j]:.{decimals}f}, {eta[i,j]:.{decimals}f})"
    return pd.DataFrame(out)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    return buf.read()


def build_pdf_report(distance_name: str, scale: int, crit_meta: pd.DataFrame, result: pd.DataFrame,
                     interpretation: str, elapsed_sec: float) -> bytes:
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    heading = styles["Heading2"]
    h1 = styles["Heading1"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    story = []
    story.append(Paragraph("PNTOPSISForge v1.0 Report", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    story.append(Paragraph(f"Primary computation time: {elapsed_sec:.6f} seconds", normal))
    story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph("Settings", heading))
    story.append(Paragraph(f"Linguistic scale: {scale}-point", normal))
    story.append(Paragraph(f"Primary distance: {distance_name}", normal))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("Criteria metadata", heading))
    meta_tbl_data = [list(crit_meta.columns)] + crit_meta.values.tolist()
    meta_tbl = Table(meta_tbl_data, hAlign="LEFT")
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph("Ranking results", heading))
    res = result.copy().reset_index().rename(columns={"index": "Alternative"})
    res_tbl_data = [list(res.columns)] + res.round(6).values.tolist()
    res_tbl = Table(res_tbl_data, hAlign="LEFT")
    res_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(res_tbl)
    story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph("Interpretation", heading))
    story.append(Paragraph(interpretation, normal))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ✅ CHANGE #1: Sample dataset download format (download → upload works)
# - It now matches the format you said works:
#   Row 1 header: Alt, C1..Cn
#   Row 2: criterion types row (B/C) under C columns
#   Row 3+: alternatives with integer crisp scores
def sample_dataset_bytes_excel(scale: int, n_criteria: int = 5) -> bytes:
    crit_names = [f"C{j+1}" for j in range(n_criteria)]

    # Balanced default: roughly half B then C (you can edit after upload)
    nB = max(1, n_criteria // 2)
    types_row = [""] + (["B"] * nB) + (["C"] * (n_criteria - nB))

    rng = np.random.default_rng(42)
    data = rng.integers(1, scale + 1, size=(5, n_criteria))
    df = pd.DataFrame(data, columns=crit_names)
    df.insert(0, "Alt", [f"A{i+1}" for i in range(5)])

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # Row 1 header is produced by df.to_excel (index=False)
        df.to_excel(writer, sheet_name="Input_Matrix", index=False, startrow=0)

        # Write criterion type row at row 2 (startrow=1), no header
        pd.DataFrame([types_row], columns=["Alt"] + crit_names).to_excel(
            writer, sheet_name="Input_Matrix", index=False, header=False, startrow=1
        )

    out.seek(0)
    return out.read()


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="PNTOPSISForge v1.0", page_icon="🧭", layout="wide")

# ✅ CHANGE #2: Fix cropped/cut-off callout box (no layout changes elsewhere)
st.markdown("""
<style>
:root{
  --blue900:#0B2E4A;
  --blue700:#0B5394;
  --blue600:#1F77B4;
  --blue100:#D7ECFF;
  --bg:#F7FAFF;
  --card:#FFFFFF;
  --border:#E6EEF8;
  --muted:#52606D;
}
.stApp{ background: var(--bg); }
html, body, [class*="css"]{ font-size:18px; color:#102A43; }
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #F2F7FF 0%, #F7FAFF 100%);
  border-right: 1px solid var(--border);
}
h1{ font-size:42px !important; color:var(--blue900) !important; margin-bottom:0.2rem; }
h2{ font-size:28px !important; color:var(--blue900) !important; }
h3{ font-size:22px !important; color:var(--blue700) !important; }
button[kind="primary"]{
  background: linear-gradient(90deg, var(--blue700), var(--blue600));
  border: 0 !important;
  font-weight: 800 !important;
  font-size: 17px !important;
  padding: 0.55rem 1.0rem !important;
}
[data-testid="stMetric"]{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
}
[data-testid="stMetricLabel"]{ color: var(--muted) !important; font-size:14px !important; }
[data-testid="stMetricValue"]{ color: var(--blue700) !important; font-size:28px !important; }

[data-testid="stDataFrame"]{ font-size:18px !important; }
[data-testid="stDataFrame"] td{ font-size:18px !important; }
[data-testid="stDataFrame"] th{
  font-size:18px !important;
  font-weight:800 !important;
  color: var(--blue900) !important;
  background-color:#EAF4FF !important;
}

/* data_editor: increase cell font (Streamlit BaseWeb grid) */
div[data-testid="stDataEditor"] *{ font-size:18px !important; }
div[data-testid="stDataEditor"] input{ font-size:18px !important; }

[data-testid="stExpander"]{
  border:1px solid var(--border);
  border-radius:14px;
  background: var(--card);
}
.block-container{ padding-top: 1.1rem; padding-bottom: 2.0rem; }

/* --- FIX: callout cropping (you showed this being cut-off) --- */
.pnt-callout{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
  overflow: visible !important;
}
.pnt-callout *{
  overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

# Header
l, r = st.columns([0.72, 0.28])
with l:
    st.title("PNTOPSISForge v1.0")
    st.caption("Conference-demo interface • Upload → Configure → Run → Export")
with r:
    # ✅ applied callout class (same layout/content, only prevents cropping)
    st.markdown(
        "<div class='pnt-callout'>"
        "<div style='font-weight:800;color:#0B2E4A;'>Tips</div>"
        "<div style='color:#52606D;font-size:14px;line-height:1.35;'>"
        "Use the sample Excel for a quick demo.<br/>Enable sensitivity to compare distances."
        "</div></div>",
        unsafe_allow_html=True,
    )

with st.expander("What this app does", expanded=False):
    st.markdown(
        """
- Converts crisp scores to **PNS triplets** using a selected linguistic scale.
- Supports **Benefit/Cost** criteria and **Equal/Manual** weights.
- Computes PIS/NIS, distances, closeness \\(P_i\\), and ranking.
- Distances: **FIQ**, **PN-Euclidean**, **PN-Hamming**.
- Exports **Excel** (full sheets) and **PDF** (summary).
        """
    )

# Sidebar
st.sidebar.header("Setup")
scale = st.sidebar.selectbox("Linguistic scale", [5, 7, 9, 11], index=2)
decimals = st.sidebar.slider("Triplet decimals", 2, 6, 2)
distance_name = st.sidebar.selectbox("Primary distance", ["FIQ", "PN-Euclidean", "PN-Hamming"], index=0)
sensitivity_on = st.sidebar.checkbox("Enable sensitivity analysis", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Data")
n_sample = st.sidebar.number_input("Sample criteria (download)", min_value=2, max_value=20, value=5, step=1)
st.sidebar.download_button(
    "⬇️ Download sample Excel",
    data=sample_dataset_bytes_excel(scale=scale, n_criteria=int(n_sample)),
    file_name=f"pntopsisforge_sample_{scale}scale_{int(n_sample)}c.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.subheader("Weights")
weight_mode = st.sidebar.radio("Criteria weights", ["Equal Weights", "Manual Weights"], index=0)

# Linguistic table
table_df = pd.DataFrame([{"Score": k, "tau": v[0], "xi": v[1], "eta": v[2]} for k, v in PNS_TABLES[scale].items()]).sort_values("Score")

# Tabs
tabs = st.tabs(["Input", "Configure", "Run & Results", "Sensitivity", "Export", "Reference"])

def read_uploaded_file(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

raw_df: Optional[pd.DataFrame] = None
with tabs[0]:
    st.subheader("Input decision matrix")
    if uploaded is not None:
        raw_df = read_uploaded_file(uploaded)
        st.success(f"Loaded: {uploaded.name}")
        st.dataframe(raw_df, use_container_width=True, height=280)
    else:
        st.info("No upload detected. Use the editor below for a quick demo.")
        m = st.number_input("Alternatives (m)", 2, 200, 5, 1)
        n = st.number_input("Criteria (n)", 2, 50, 5, 1)
        grid = pd.DataFrame(np.ones((int(m), int(n)), dtype=int), columns=[f"C{j+1}" for j in range(int(n))])
        raw_df = st.data_editor(grid, use_container_width=True, height=280, key="manual_grid")

    if raw_df is None or raw_df.shape[0] == 0:
        st.stop()

# Parse input (shared)
df = raw_df.copy()
first_col_values = df.iloc[:, 0].tolist()
first_col_is_alt = False
try:
    _ = [int(str(v).strip()) for v in first_col_values[: min(10, len(first_col_values))]]
except Exception:
    first_col_is_alt = True

if first_col_is_alt:
    alt_names = df.iloc[:, 0].astype(str).tolist()
    df_mat = df.iloc[:, 1:].copy()
else:
    alt_names = [f"A{i+1}" for i in range(df.shape[0])]
    df_mat = df.copy()

# Detect B/C row
first_row = df_mat.iloc[0, :].tolist()
has_bc = is_bc_row(first_row)
if has_bc:
    crit_types = [str(x).strip().upper() for x in first_row]
    df_scores = df_mat.iloc[1:, :].copy()
    alt_names = alt_names[1:]
else:
    crit_types = ["B"] * df_mat.shape[1]
    df_scores = df_mat.copy()

df_scores = df_scores.loc[:, ~df_scores.columns.astype(str).str.contains("^Unnamed")].copy()
crit_names = [str(c) for c in df_scores.columns]

# Validate crisp
validation_error = None
try:
    crisp_int = coerce_int_matrix(df_scores.reset_index(drop=True))
    validate_score_range(crisp_int, scale)
except Exception as e:
    validation_error = str(e)

# Configure tab
with tabs[1]:
    st.subheader("Configure criteria & weights")
    if has_bc:
        st.info("Detected a criterion type row (B/C) in the first row under criteria.")
    else:
        st.warning("No criterion type row detected. Default is all Benefit (B). Please adjust if needed.")

    type_df = pd.DataFrame([crit_types], columns=crit_names, index=["Type (B/C)"])
    edited_type_df = st.data_editor(type_df, use_container_width=True, key="crit_types_editor")
    crit_types = [str(edited_type_df.iloc[0, j]).strip().upper() for j in range(len(crit_names))]
    if any(t not in {"B", "C"} for t in crit_types):
        st.error("Criterion types must be only 'B' or 'C'.")
        st.stop()

    if validation_error:
        st.error(f"Fix input first: {validation_error}")
        st.stop()

    crisp_show = crisp_int.copy()
    crisp_show.index = alt_names
    st.markdown("#### Crisp matrix (validated)")
    st.dataframe(crisp_show, use_container_width=True, height=280)

    st.markdown("#### Weights")
    n_criteria = len(crit_names)
    if weight_mode == "Equal Weights":
        w = np.array([1.0 / n_criteria] * n_criteria, dtype=float)
        st.success(f"Equal weights: each w_j = 1/{n_criteria} = {1.0/n_criteria:.6f}")
    else:
        w_default = pd.DataFrame([[round(1.0 / n_criteria, 6)] * n_criteria], columns=crit_names, index=["w"])
        w_edit = st.data_editor(w_default, use_container_width=True, key="weights_editor")
        w = np.array([float(w_edit.iloc[0, j]) for j in range(n_criteria)], dtype=float)
        w_sum = float(np.sum(w))
        if abs(w_sum - 1.0) > 1e-3:
            st.warning(f"Sum of weights = {w_sum:.6f} (recommended 1.000000).")
        else:
            st.success("Weights sum to 1.000000 ✅")

# Run tab
with tabs[2]:
    st.subheader("Run & Results")
    if validation_error:
        st.error(f"Fix input first: {validation_error}")
        st.stop()

    run = st.button("🚀 Run PNTOPSISForge", type="primary")

    if "computed" not in st.session_state:
        st.session_state["computed"] = False

    if run:
        t0 = time.perf_counter()
        tau, xi, eta = map_crisp_to_pns(crisp_int, scale)
        tau_n, xi_n, eta_n = normalize_pns(tau, xi, eta, crit_types)
        tau_w, xi_w, eta_w = apply_weights(tau_n, xi_n, eta_n, w)
        tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg = compute_ideals(tau_w, xi_w, eta_w, crit_types)
        t1 = time.perf_counter()
        primary_result = compute_result_for_distance(distance_name, tau_w, xi_w, eta_w, tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg, alt_names)
        t2 = time.perf_counter()

        st.session_state["computed"] = True
        st.session_state["payload"] = dict(
            tau=tau, xi=xi, eta=eta,
            tau_n=tau_n, xi_n=xi_n, eta_n=eta_n,
            tau_w=tau_w, xi_w=xi_w, eta_w=eta_w,
            tau_p=tau_p, xi_p=xi_p, eta_p=eta_p,
            tau_neg=tau_neg, xi_neg=xi_neg, eta_neg=eta_neg,
            primary_result=primary_result,
            preprocess_sec=t1 - t0,
            primary_sec=t2 - t1,
        )

    if not st.session_state["computed"]:
        st.info("Click **Run PNTOPSISForge** to compute.")
        st.stop()

    P = st.session_state["payload"]
    primary_result = P["primary_result"]
    best_alt = primary_result.index[0]
    best_pi = float(primary_result.iloc[0]["P_i"])
    primary_sec = P["primary_sec"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary distance", distance_name)
    c2.metric("Best alternative", best_alt)
    c3.metric("Best P_i", f"{best_pi:.6f}")
    c4.metric("Runtime", f"{primary_sec:.6f} s")

    st.markdown("#### Ranking table")
    def _highlight(row):
        if row.name == best_alt:
            return ["background-color: #D7ECFF; font-weight: 800"] * len(row)
        return [""] * len(row)
    st.dataframe(primary_result.style.apply(_highlight, axis=1), use_container_width=True, height=280)

    st.markdown("#### Closeness chart (P_i)")
    pi_df = primary_result.reset_index().rename(columns={"index": "Alternative"})
    chart = alt.Chart(pi_df).mark_bar().encode(
        x=alt.X("Alternative:N", sort=None),
        y=alt.Y("P_i:Q"),
        color=alt.condition(alt.datum.Alternative == best_alt, alt.value("#0B5394"), alt.value("#1F77B4")),
        tooltip=["Alternative:N", alt.Tooltip("P_i:Q", format=".6f")],
    ).properties(height=340)
    st.altair_chart(chart.configure_view(strokeWidth=0).configure_axis(labelFontSize=14, titleFontSize=16), use_container_width=True)

    # ✅ CHANGE #3: Summary wording (refined) - your preferred structure + best version
    best_s_plus = float(primary_result.iloc[0]["S_i_plus"])
    best_s_minus = float(primary_result.iloc[0]["S_i_minus"])
    st.markdown("#### Summary")
    st.write(
        f"{best_alt} is the best alternative because it has the highest relative closeness "
        f"P_i = {best_pi:.6f}. Under {distance_name}, it achieves a smaller distance to the positive ideal "
        f"(S_i_plus = {best_s_plus:.6f}) and a larger distance from the negative ideal "
        f"(S_i_minus = {best_s_minus:.6f})."
    )

# Sensitivity tab
with tabs[3]:
    st.subheader("Sensitivity analysis")
    if not st.session_state.get("computed", False):
        st.info("Run the model first (Run & Results tab).")
        st.stop()

    if not sensitivity_on:
        st.info("Enable sensitivity analysis in the sidebar.")
        st.stop()

    P = st.session_state["payload"]
    tau_w, xi_w, eta_w = P["tau_w"], P["xi_w"], P["eta_w"]
    tau_p, xi_p, eta_p = P["tau_p"], P["xi_p"], P["eta_p"]
    tau_neg, xi_neg, eta_neg = P["tau_neg"], P["xi_neg"], P["eta_neg"]
    primary_result = P["primary_result"]

    t3 = time.perf_counter()
    frames = {}
    for dname in ["PN-Euclidean", "PN-Hamming", "FIQ"]:
        frames[dname] = compute_result_for_distance(dname, tau_w, xi_w, eta_w, tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg, alt_names)[["P_i", "Rank"]]
        frames[dname] = frames[dname].rename(columns={"P_i": f"P_i ({dname})", "Rank": f"Rank ({dname})"})
    compare = pd.concat([frames["PN-Euclidean"], frames["PN-Hamming"], frames["FIQ"]], axis=1)
    compare = compare.loc[primary_result.index]
    t4 = time.perf_counter()
    st.success(f"Computed in {t4 - t3:.6f} s")
    st.dataframe(compare, use_container_width=True, height=280)

# Export tab
with tabs[4]:
    st.subheader("Export")
    if not st.session_state.get("computed", False):
        st.info("Run the model first (Run & Results tab).")
        st.stop()

    P = st.session_state["payload"]
    primary_result = P["primary_result"]
    primary_sec = P["primary_sec"]

    tau, xi, eta = P["tau"], P["xi"], P["eta"]
    tau_n, xi_n, eta_n = P["tau_n"], P["xi_n"], P["eta_n"]
    tau_w, xi_w, eta_w = P["tau_w"], P["xi_w"], P["eta_w"]
    tau_p, xi_p, eta_p = P["tau_p"], P["xi_p"], P["eta_p"]
    tau_neg, xi_neg, eta_neg = P["tau_neg"], P["xi_neg"], P["eta_neg"]

    pns_df = format_triplets(tau, xi, eta, decimals=decimals); pns_df.columns = crit_names; pns_df.index = alt_names
    norm_df = format_triplets(tau_n, xi_n, eta_n, decimals=decimals); norm_df.columns = crit_names; norm_df.index = alt_names
    w_df2 = format_triplets(tau_w, xi_w, eta_w, decimals=decimals); w_df2.columns = crit_names; w_df2.index = alt_names

    pis = pd.DataFrame({"tau+": tau_p, "xi+": xi_p, "eta+": eta_p}, index=crit_names)
    nis = pd.DataFrame({"tau-": tau_neg, "xi-": xi_neg, "eta-": eta_neg}, index=crit_names)
    pisnis = pd.concat([pis, nis], axis=1)

    crisp_show = crisp_int.copy(); crisp_show.index = alt_names
    meta = pd.DataFrame({"Criterion": crit_names, "Type (B/C)": crit_types, "Weight": w})

    sheets = {
        "Crisp_Matrix": crisp_show,
        "PNS_Matrix": pns_df,
        "Normalized": norm_df,
        "Weighted_Normalized": w_df2,
        "PIS_NIS": pisnis,
        "Results_Primary": primary_result,
        "Meta": meta.set_index("Criterion"),
        "Linguistic_Table": table_df.set_index("Score"),
    }
    xlsx_bytes = to_excel_bytes(sheets)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download results (Excel)",
            data=xlsx_bytes,
            file_name="pntopsisforge_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # keep your existing short interpretation for PDF
    best_alt = primary_result.index[0]
    best_pi = float(primary_result.iloc[0]["P_i"])
    interpretation = f"{best_alt} is ranked #1 with P_i = {best_pi:.6f} under {distance_name}."
    pdf_bytes = build_pdf_report(distance_name=distance_name, scale=scale, crit_meta=meta, result=primary_result, interpretation=interpretation, elapsed_sec=primary_sec)
    with col2:
        st.download_button(
            "⬇️ Download report (PDF)",
            data=pdf_bytes,
            file_name="pntopsisforge_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# Reference tab
with tabs[5]:
    st.subheader("Reference: Linguistic table")
    st.dataframe(table_df, use_container_width=True, height=300)