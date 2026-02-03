import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# --- 1. CONFIGURAZIONE E STILE ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root { --primary: #1e3a8a; --bg: #f8fafc; }

[data-testid="stAppViewContainer"] {
  background-color: #f8fafc;
  color: #1e3a8a;
}

html, body, .stApp {
  font-family: 'Inter', sans-serif;
  background-color: #f8fafc;
}

section[data-testid="stSidebar"] {
  background-color: #ffffff !important;
  border-right: 1px solid #e2e8f0;
}

.settings-title {
  font-size: 20px;
  font-weight: 700;
  color: #1e3a8a;
  margin-bottom: 15px;
}

.blue-section-header {
  background-color: #1e3a8a;
  padding: 10px 15px;
  border-radius: 8px;
  margin-top: 20px;
  margin-bottom: 10px;
}

.blue-section-header p {
  color: #ffffff !important;
  margin: 0 !important;
  font-weight: 700 !important;
  font-size: 15px !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"],
.custom-summary-box {
  background-color: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
}

input, span, .custom-summary-box p {
  color: #1e3a8a !important;
  font-weight: 600;
}

div[data-baseweb="input"] button {
  background-color: #f1f5f9 !important;
  color: #1e3a8a !important;
}

section[data-testid="stSidebar"] label {
  color: #1e3a8a !important;
  font-weight: 700;
}

/* main area cards */
div[data-testid="stVerticalBlock"] > div {
  background-color: white !important;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* --- Sidebar header: Settings + Guide --- */
.sidebar-header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:10px;
}

.sidebar-title{
  font-size:20px;
  font-weight:700;
  color:#1e3a8a;
}

/* IMPORTANT: disable card styling INSIDE sidebar only */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PROXY SETTINGS
# ============================================================
NON_MINED = {"H", "N", "O"}
BIG = 1e30
REE = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Y".split())
WORLD_REO_PROD_2024 = 390000.0
WORLD_REO_RESERVES  = 90_000_000.0

# --- 2. MOTORE DI CARICAMENTO E PULIZIA ---
def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^-0-9.]', '', regex=True), errors='coerce')

def _find_col_by_keyword(df: pd.DataFrame, keyword: str) -> str:
    hits = [c for c in df.columns if keyword.lower() in str(c).lower()]
    if not hits:
        raise ValueError(f"Non trovo una colonna che contenga keyword '{keyword}'. Colonne disponibili: {list(df.columns)[:20]} ...")
    return hits[0]

def _build_prop_vector(db: pd.DataFrame, col_keyword: str) -> np.ndarray:
    col_name = _find_col_by_keyword(db, col_keyword)
    v = clean_numeric(db[col_name])
    v = pd.Series(v.values, index=db["Z"].values).reindex(range(1, 119))
    return v.to_numpy(dtype=float)

def _weakest_link_vectorized(af_matrix: np.ndarray, v_elem: np.ndarray) -> np.ndarray:
    x = np.asarray(af_matrix, dtype=float)
    v = np.asarray(v_elem, dtype=float).reshape(1, -1)
    ratio = np.where(x > 0, v / np.maximum(x, 1e-30), np.inf)
    used = (x > 0)
    has_nan_used = np.any(used & np.isnan(v), axis=1)
    out = np.min(ratio, axis=1)
    out = np.where(has_nan_used, np.nan, out)
    return out

def _bottleneck_info(af_matrix: np.ndarray, v_elem: np.ndarray, elem_symbols_by_Z: dict | None):
    x = np.asarray(af_matrix, dtype=float)
    v = np.asarray(v_elem, dtype=float).reshape(1, -1)
    ratio = np.where(x > 0, v / np.maximum(x, 1e-30), np.inf)
    used = x > 0
    has_nan_used = np.any(used & np.isnan(v), axis=1)
    ratio_safe = np.where(np.isnan(ratio), np.inf, ratio)
    sorted_ratio = np.sort(ratio_safe, axis=1)
    min1 = sorted_ratio[:, 0]
    min2 = sorted_ratio[:, 1] 
    argmin = np.argmin(ratio_safe, axis=1) 
    Z_lim = (argmin + 1).astype(int)
    if elem_symbols_by_Z:
        bottleneck_symbol = np.array([elem_symbols_by_Z.get(int(z), f"Z{int(z)}") for z in Z_lim], dtype=object)
    else:
        bottleneck_symbol = np.array([f"Z{int(z)}" for z in Z_lim], dtype=object)
    bottleneck_ratio = min2 / np.maximum(min1, 1e-30)
    bottleneck_symbol = np.where(has_nan_used, None, bottleneck_symbol)
    min1 = np.where(has_nan_used, np.nan, min1)
    min2 = np.where(has_nan_used, np.nan, min2)
    bottleneck_ratio = np.where(has_nan_used, np.nan, bottleneck_ratio)
    return bottleneck_symbol, min1, min2, bottleneck_ratio

def _weighted_avg_with_nan_propagation(af_matrix: np.ndarray, v_elem: np.ndarray) -> np.ndarray:
    x = np.asarray(af_matrix, dtype=float)
    v = np.asarray(v_elem, dtype=float).reshape(1, -1)
    used = x > 0
    has_nan_used = np.any(used & np.isnan(v), axis=1)
    out = x @ np.nan_to_num(v_elem, nan=0.0)
    out = np.where(has_nan_used, np.nan, out)
    return out

def _apply_proxies(v_prod: np.ndarray, v_res: np.ndarray, elem_symbols_by_Z: dict | None) -> tuple[np.ndarray, np.ndarray, dict]:
    info = {"ree_missing_prod": [], "ree_missing_res": [], "note": ""}
    vp = v_prod.copy().astype(float)
    vr = v_res.copy().astype(float)
    if not elem_symbols_by_Z:
        floor_p = np.nanmin(vp); floor_r = np.nanmin(vr)
        vp = np.where(np.isnan(vp), floor_p, vp)
        vr = np.where(np.isnan(vr), floor_r, vr)
        return vp, vr, info
    Zs = np.arange(1, 119)
    nm_mask = np.array([elem_symbols_by_Z.get(int(z), "").strip() in NON_MINED for z in Zs])
    vp[nm_mask] = np.where(np.isnan(vp[nm_mask]), BIG, vp[nm_mask])
    vr[nm_mask] = np.where(np.isnan(vr[nm_mask]), BIG, vr[nm_mask])
    ree_mask = np.array([elem_symbols_by_Z.get(int(z), "").strip() in REE for z in Zs])
    known_prod = np.nansum(vp[ree_mask])
    missing_prod_idx = np.where(ree_mask & np.isnan(vp))[0]
    rem_prod = max(WORLD_REO_PROD_2024 - known_prod, 0.0)
    if len(missing_prod_idx) > 0:
        fill_prod = rem_prod / len(missing_prod_idx)
        vp[missing_prod_idx] = fill_prod
        info["ree_missing_prod"] = [elem_symbols_by_Z.get(int(i+1), f"Z{i+1}") for i in missing_prod_idx]
    known_res = np.nansum(vr[ree_mask])
    missing_res_idx = np.where(ree_mask & np.isnan(vr))[0]
    rem_res = max(WORLD_REO_RESERVES - known_res, 0.0)
    if len(missing_res_idx) > 0:
        fill_res = rem_res / len(missing_res_idx)
        vr[missing_res_idx] = fill_res
        info["ree_missing_res"] = [elem_symbols_by_Z.get(int(i+1), f"Z{i+1}") for i in missing_res_idx]
    floor_p = np.nanmin(vp); floor_r = np.nanmin(vr)
    vp = np.where(np.isnan(vp), floor_p, vp)
    vr = np.where(np.isnan(vr), floor_r, vr)
    return vp, vr, info

@st.cache_data
def load_and_sync_data():
    df = pd.read_csv("AF_vectors.csv")
    db = pd.read_csv("Materials Database 1.csv")
    sus = pd.read_csv("MF_sustainability_rank.csv")
    
    sus.columns = [str(c).strip() for c in sus.columns]
    df.columns  = [str(c).strip() for c in df.columns]
    
    S_cols = []
    for i in range(1, 11):
        target = f"S{i}"
        if target in sus.columns:
            S_cols.append(target)
        else:
            hits = [c for c in sus.columns if c.upper().startswith(target)]
            if hits: S_cols.append(hits[0])
            else: raise ValueError(f"MF_sustainability_rank.csv: cannot find a column for {target}.")
    
    join_key = None
    if "Original_Index" in df.columns and "Original_Index" in sus.columns: join_key = "Original_Index"
    elif "Material_Name" in df.columns and "Material_Name" in sus.columns: join_key = "Material_Name"
    else: raise ValueError("No common key to merge sustainability scores.")
    
    new_metrics_map = {
        "Compound_CO2_footprint_with_estimated_recycling_rate_CO2_per_kg": "CO2/kg rec.",
        "Compound_CO2_footprint_kg_CO2_per_kg": "CO2/kg",
        "Compound_Energy_footprint_MJ_per_kg": "MJ/kg",
        "Compound_Energy_footprint_with_estimated_recycling_rate_MJ_per_kg": "MJ/kg rec.",
        "Compound_Water_usage_l_per_kg": "L/kg"
    }
    cols_to_merge = [join_key] + S_cols
    valid_new_cols = []
    for csv_col in new_metrics_map.keys():
        if csv_col in sus.columns: valid_new_cols.append(csv_col)
    cols_to_merge += valid_new_cols
    
    sus_small = sus[cols_to_merge].copy()
    df = df.merge(sus_small, on=join_key, how="left")
    
    rename_map = {S_cols[i-1]: f"S{i}" for i in range(1, 11)}
    for csv_col in valid_new_cols: rename_map[csv_col] = new_metrics_map[csv_col]
    df = df.rename(columns=rename_map)

    if "Z" not in db.columns: raise ValueError("Nel database elementi manca la colonna 'Z'.")
    db = db.dropna(subset=["Z"]).copy()
    db["Z"] = pd.to_numeric(db["Z"], errors="coerce").astype(int)

    elem_col = None
    for c in db.columns:
        if "element" in str(c).lower():
            elem_col = c; break
    elem_symbols_by_Z = None
    if elem_col is not None:
        tmp = db[["Z", elem_col]].dropna()
        elem_symbols_by_Z = {int(z): str(sym).strip() for z, sym in zip(tmp["Z"], tmp[elem_col])}

    v_prod_raw = _build_prop_vector(db, "production")
    v_res_raw  = _build_prop_vector(db, "reserve")
    v_prod, v_res, proxy_info = _apply_proxies(v_prod_raw, v_res_raw, elem_symbols_by_Z)

    af_cols = [f"AF_{i}" for i in range(1, 119)]
    af_matrix = df[af_cols].fillna(0.0).to_numpy(dtype=float)

    df["Pmax_t_per_yr"] = _weakest_link_vectorized(af_matrix, v_prod)
    df["Plong_t"]       = _weakest_link_vectorized(af_matrix, v_res)

    bn_el_P, bn_min1_P, bn_min2_P, bn_ratio_P = _bottleneck_info(af_matrix, v_prod, elem_symbols_by_Z)
    df["Bottleneck_prod_element"] = bn_el_P
    df["Bottleneck_prod_min1"] = bn_min1_P
    df["Bottleneck_prod_min2"] = bn_min2_P
    df["Bottleneck_prod_ratio"] = bn_ratio_P
    
    bn_el_R, bn_min1_R, bn_min2_R, bn_ratio_R = _bottleneck_info(af_matrix, v_res, elem_symbols_by_Z)
    df["Bottleneck_res_element"] = bn_el_R
    df["Bottleneck_res_min1"] = bn_min1_R
    df["Bottleneck_res_min2"] = bn_min2_R
    df["Bottleneck_res_ratio"] = bn_ratio_R
    
    try:
        v_hhi = _build_prop_vector(db, "HHI"); v_esg = _build_prop_vector(db, "ESG")
        v_sr = _build_prop_vector(db, "Supply risk"); v_comp = _build_prop_vector(db, "Companionality (%)")
        df["HHI"] = _weighted_avg_with_nan_propagation(af_matrix, v_hhi)
        df["ESG"] = _weighted_avg_with_nan_propagation(af_matrix, v_esg)
        df["Supply risk"] = _weighted_avg_with_nan_propagation(af_matrix, v_sr)
        df["Companionality (%)"] = _weighted_avg_with_nan_propagation(af_matrix, v_comp)
    except Exception: pass

    for label in new_metrics_map.values():
        if label in df.columns: df[label] = pd.to_numeric(df[label], errors="coerce")
    for c in ["P1", "P2", "P3"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df.attrs["proxy_info"] = proxy_info
    df.attrs["has_elem_symbols"] = bool(elem_symbols_by_Z)
    return df

# --- 3. MOTORE DI CALCOLO RANKING ---
def generate_linear_scores(n_tiers):
    return [round((i + 1) / n_tiers, 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, n_tiers, thresholds):
    scores = generate_linear_scores(n_tiers)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    if not thresholds: return assigned
    for i in range(len(thresholds)):
        assigned[df[col_name] >= thresholds[i]] = scores[i+1]
    return assigned

def check_ascending_order(values, label):
    """Verifica che i valori siano strettamente crescenti"""
    if not values: return
    # Controllo se ogni elemento è minore del successivo
    if not all(x < y for x, y in zip(values, values[1:])):
        st.error(f"❌ Error in **{label}**: Values must be in strict ascending order (e.g., 100 < 200 < 300).")
        st.stop() # Blocca l'esecuzione dell'app

# --- 4. INTERFACCIA APP ---
df = load_and_sync_data().copy()

manual_thresholds = {"P1": [], "P2": [], "P3": []}

all_metrics_options = ["SS", "HHI", "ESG", "Supply risk", "Companionality (%)", 
                       "CO2/kg", "CO2/kg rec.", "MJ/kg", "MJ/kg rec.", "L/kg"]

metric_descriptions = {
    "SS": "Sustainability Score.",
    "HHI": "Herfindahl-Hirschman Index.",
    "ESG": "Environmental, Social, and Governance score.",
    "Supply risk": "Risk associated with supply chain.",
    "Companionality (%)": "% produced as byproduct.",
    "CO2/kg": "Carbon Footprint.",
    "CO2/kg rec.": "Carbon Footprint (with recycling).",
    "MJ/kg": "Energy Consumption.",
    "MJ/kg rec.": "Energy Consumption (with recycling).",
    "L/kg": "Water Footprint."
}

with st.sidebar:
    st.markdown("""<div class="sidebar-header"><div class="sidebar-title">Settings</div></div>""", unsafe_allow_html=True)

    if st.button("♻️ Clear Cache & Reload"):
        st.cache_data.clear()
        st.rerun()

    GUIDE_PATH = "GreenNanoAnalyticsGuide.pdf"
    if os.path.exists(GUIDE_PATH):
        with open(GUIDE_PATH, "rb") as f:
            st.download_button("📘", f, "GreenNanoAnalyticsGuide.pdf", "application/pdf")

    # --- INPUT THRESHOLDS + VALIDATION ---
    
    # --- P1 ---
    sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2, key="sf_P1")
    sc_t = generate_linear_scores(sf_t)
    for i in range(sf_t - 1):
        val = st.number_input(f"Threshold Score {sc_t[i+1]} (P1)", value=int(350 + (i * 50)), step=1, key=f"p1_{i}")
        manual_thresholds["P1"].append(float(val))
    
    # SAFETY CHECK P1
    check_ascending_order(manual_thresholds["P1"], "P1 (Temperature)")

    # --- P2 & P3 ---
    for label, key, d_idx, d_val in [("Magnetization (T)", "P2", 1, 0.4), ("Coercivity (T)", "P3", 3, 0.4)]:
        st.markdown(f"**{label}**")
        sf = st.selectbox(f"Subcategories ({key})", [2, 3, 4, 5], index=d_idx, key=f"sf_{key}")
        sc = generate_linear_scores(sf)
        for i in range(sf - 1):
            v = st.number_input(f"Threshold Score {sc[i+1]} ({key})", value=float(d_val + (i * 0.2)), key=f"t_{key}_{i}")
            manual_thresholds[key].append(float(v))
        
        # SAFETY CHECK P2/P3
        check_ascending_order(manual_thresholds[key], f"{key} ({label})")

        if key == "P2": sf_m = sf
        else: sf_c = sf

    # --- WEIGHTS ---
    st.markdown('<div class="blue-section-header"><p>2. Performance Weights</p></div>', unsafe_allow_html=True)
    w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33, key="w_p1")
    rem = float(round(1.0 - w_p1, 2))
    w_p2 = st.slider("Weight P2 (Mag)", 0.0, rem, min(0.33, rem), key="w_p2")
    w_p3 = float(round(max(0.0, 1.0 - (w_p1 + w_p2)), 2))
    st.info(f"Weight P3: {w_p3:.2f}")

    # --- PARETO SETTINGS ---
    st.markdown('<div class="blue-section-header"><p>Pareto settings (ε)</p></div>', unsafe_allow_html=True)
    eps_ops = st.slider("ε on OPS", 0.0, 0.60, 0.20, 0.02)
    eps_ss = st.slider("ε on SS", 0.0, 0.40, 0.05, 0.01)

    # --- SCALABILITY ---
    st.markdown('<div class="blue-section-header"><p>3. Scalability View</p></div>', unsafe_allow_html=True)
    color_metric = st.selectbox("Coloring Metric", all_metrics_options, index=0)
    
    # --- SS WEIGHTS ---
    st.markdown('<div class="blue-section-header"><p>3B. Sustainability Weights</p></div>', unsafe_allow_html=True)
    default_w = [0.1] * 10
    w_in = []
    for i in range(1, 11):
        w_in.append(st.number_input(f"Weight S{i}", 0.0, 1.0, default_w[i-1], 0.01, key=f"w_s{i}"))
    w_sum = float(np.sum(w_in))
    if abs(w_sum - 1.0) > 1e-6:
        st.error(f"❌ Sum must be 1. Current: {w_sum:.2f}")
        st.stop()
    w_ss = np.array(w_in, dtype=float)

    # --- TREND ---
    st.markdown('<div class="blue-section-header"><p>4. Top-right Trend</p></div>', unsafe_allow_html=True)
    trend_metrics = st.multiselect("Metrics vs H", all_metrics_options, default=all_metrics_options)

# --- CALCOLO SCORE (Fuori da sidebar) ---
S_cols = [f"S{i}" for i in range(1, 11)]
if any(c not in df.columns for c in S_cols):
    st.error("Missing Sustainability columns.")
    st.stop()

S = df[S_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
S = np.where(np.isnan(S), 0.3, S)
S = np.clip(S, 1e-12, 1.0)
df["SS"] = np.exp((np.log(S) * w_ss.reshape(1, -1)).sum(axis=1))

p1_s = assign_tiered_scores(df, "P1", sf_t, manual_thresholds["P1"]) if "P1" in df.columns else 1.0
p2_s = assign_tiered_scores(df, "P2", sf_m, manual_thresholds["P2"]) if "P2" in df.columns else 1.0
p3_s = assign_tiered_scores(df, "P3", sf_c, manual_thresholds["P3"]) if "P3" in df.columns else 1.0

df["OPS"] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)

# --- PARETO CALCULATION ---
pts = df[["OPS", "SS"]].to_numpy(dtype=float)
efficient = np.ones(pts.shape[0], dtype=bool)
for i, c in enumerate(pts):
    if efficient[i]:
        efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))

if np.any(efficient) and (eps_ops > 0 or eps_ss > 0):
    pareto_vals = df.loc[efficient, ["OPS", "SS"]].values
    def is_close_enough(row):
        return np.any((row["OPS"] >= (1 - eps_ops) * pareto_vals[:, 0]) & 
                      (row["SS"] >= (1 - eps_ss) * pareto_vals[:, 1]))
    efficient_soft = df.apply(is_close_enough, axis=1)
else:
    efficient_soft = efficient

df["Status"] = np.where(efficient_soft, "Optimal Choice", "Standard")

# --- TABS E VISUALIZZAZIONE ---
t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "📈 Top-right Trend"])

with t1:
    colA, colB = st.columns([2, 1])
    
    with colA:
        fig = px.scatter(
            df, x="OPS", y="SS", color="Status",
            hover_name="Material_Name" if "Material_Name" in df.columns else None,
            color_discrete_map={"Optimal Choice": "#1e3a8a", "Standard": "#cbd5e1"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("**Top Pareto Materials**")
        show_cols = [c for c in ["Material_Name", "OPS", "SS"] if c in df.columns]
        
        # Chiave univoca per forzare il refresh
        unique_key = f"table_{w_p1}_{w_p2}_{eps_ops}_{eps_ss}_{len(manual_thresholds['P1'])}"
        
        st.dataframe(
            df[efficient_soft].sort_values(by="OPS", ascending=False)[show_cols],
            use_container_width=True,
            height=500,
            key=unique_key 
        )
        
        if "Bottleneck_prod_element" in df.columns:
            counts = df.loc[efficient_soft, "Bottleneck_prod_element"].value_counts().head(5)
            if len(counts) > 0:
                st.markdown("**Most frequent production bottlenecks (Pareto set)**")
                for el, n in counts.items():
                    st.write(f"• **{el}** → {n} materials")

with t2:
    st.markdown("### Scalability (Weakest-link)")
    df_plot = df.dropna(subset=["Pmax_t_per_yr", "Plong_t"]).copy()
    
    metric_col = color_metric
    if metric_col not in df_plot.columns: metric_col = "SS"
    
    df_nonan = df_plot[df_plot[metric_col].notna()]
    df_nan = df_plot[df_plot[metric_col].isna()]
    
    fig_sc = go.Figure()
    if not df_nonan.empty:
        fig_sc.add_trace(go.Scatter(
            x=df_nonan["Plong_t"], y=df_nonan["Pmax_t_per_yr"], mode="markers",
            marker=dict(
                size=np.where(df_nonan["Status"] == "Optimal Choice", 10, 7),
                color=df_nonan[metric_col], colorscale="Viridis", showscale=True,
                colorbar=dict(title=metric_col), opacity=0.9
            ),
            text=df_nonan["Material_Name"],
            hovertemplate="%{text}<br>Plong=%{x:.3g}<br>Pmax=%{y:.3g}<br>Val=%{marker.color:.3g}"
        ))
    if not df_nan.empty:
        fig_sc.add_trace(go.Scatter(
            x=df_nan["Plong_t"], y=df_nan["Pmax_t_per_yr"], mode="markers",
            marker=dict(size=7, color="lightgrey", opacity=0.9),
            text=df_nan["Material_Name"], hovertemplate="%{text}<br>Val=NaN"
        ))
    
    fig_sc.update_layout(template="plotly_white", height=600, xaxis_type="log", yaxis_type="log",
                         xaxis_title="Long-term prod (tons)", yaxis_title="Max yearly prod (t/yr)")
    st.plotly_chart(fig_sc, use_container_width=True)

with t3:
    st.markdown("### Does a metric increase when moving top-right?")
    base = df.dropna(subset=["Pmax_t_per_yr", "Plong_t"]).copy()
    base = base[(base["Pmax_t_per_yr"] > 0) & (base["Plong_t"] > 0)]
    
    if not base.empty:
        base["_H_"] = np.log(base["Pmax_t_per_yr"]) + np.log(base["Plong_t"])
        cols = st.columns(2)
        idx = 0
        for metric in trend_metrics:
            if metric not in base.columns: continue
            tmp = base.dropna(subset=[metric])
            if len(tmp) < 3: continue
            
            x = tmp["_H_"]; y = tmp[metric]
            m, q = np.polyfit(x, y, 1)
            r2 = np.corrcoef(x, y)[0,1]**2
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=metric))
            fig.add_trace(go.Scatter(x=x, y=m*x+q, mode="lines", name="Trend"))
            fig.update_layout(template="plotly_white", height=350, title=f"{metric} (R²={r2:.3f})",
                              xaxis_title="H = log(Pmax)+log(Plong)", yaxis_title=metric)
            
            with cols[idx % 2]:
                st.plotly_chart(fig, use_container_width=True)
            idx += 1
