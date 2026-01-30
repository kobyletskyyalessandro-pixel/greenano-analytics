import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# --- 1. CONFIGURAZIONE E STILE ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root { --primary: #1e3a8a; --bg: #f8fafc; }
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e3a8a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .settings-title { font-size: 20px; font-weight: 700; color: #1e3a8a; margin-bottom: 15px; }
    .blue-section-header { background-color: #1e3a8a; padding: 10px 15px; border-radius: 8px; margin-top: 20px; margin-bottom: 10px; }
    .blue-section-header p { color: #ffffff !important; margin: 0 !important; font-weight: 700 !important; font-size: 15px !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"], .custom-summary-box {
        background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    }
    input, span, .custom-summary-box p { color: #1e3a8a !important; font-weight: 600; }
    div[data-baseweb="input"] button { background-color: #f1f5f9 !important; color: #1e3a8a !important; }
    section[data-testid="stSidebar"] label { color: #1e3a8a !important; font-weight: 700; }
    div[data-testid="stVerticalBlock"] > div { background-color: white !important; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# PROXY SETTINGS (come il tuo script)
# ============================================================
NON_MINED = {"H", "N", "O"}
BIG = 1e30
REE = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Y".split())
WORLD_REO_PROD_2024 = 390000.0
WORLD_REO_RESERVES  = 90_000_000.0

# --- 2. MOTORE DI CARICAMENTO E PULIZIA ---
def clean_numeric(series):
    """Pulisce stringhe con virgole, testi (es. 'primary') e converte in numeri."""
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





def _weighted_avg_with_nan_propagation(af_matrix: np.ndarray, v_elem: np.ndarray) -> np.ndarray:
    """
    Media pesata per materiale: sum_i x_i * v_i
    Se un materiale usa un elemento con v_i = NaN -> risultato = NaN
    (così nel plot viene colorato in grigio)
    """
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
        floor_p = np.nanmin(vp)
        floor_r = np.nanmin(vr)
        vp = np.where(np.isnan(vp), floor_p, vp)
        vr = np.where(np.isnan(vr), floor_r, vr)
        info["note"] = "No element symbols found -> only floor proxy applied."
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

    floor_p = np.nanmin(vp)
    floor_r = np.nanmin(vr)
    vp = np.where(np.isnan(vp), floor_p, vp)
    vr = np.where(np.isnan(vr), floor_r, vr)

    return vp, vr, info

@st.cache_data
def load_and_sync_data():
    # --- carica file ---
    df = pd.read_csv("AF_vectors.csv")
    db = pd.read_csv("Materials Database 1.csv")
    # --- bring S1..S10 from MF_sustainability_rank.csv (material-level file) ---
    sus = pd.read_csv("MF_sustainability_rank.csv")
    
    # normalize column names (strip spaces)
    sus.columns = [str(c).strip() for c in sus.columns]
    df.columns  = [str(c).strip() for c in df.columns]
    
    # find S1..S10 columns robustly (accept exact "S1" or things that start with "S1")
    S_cols = []
    for i in range(1, 11):
        target = f"S{i}"
        if target in sus.columns:
            S_cols.append(target)
        else:
            # fallback: first column whose name starts with "S{i}"
            hits = [c for c in sus.columns if c.upper().startswith(target)]
            if hits:
                S_cols.append(hits[0])
            else:
                raise ValueError(f"MF_sustainability_rank.csv: cannot find a column for {target}.")
    
    # choose join key (prefer Original_Index)
    join_key = None
    if "Original_Index" in df.columns and "Original_Index" in sus.columns:
        join_key = "Original_Index"
    elif "Material_Name" in df.columns and "Material_Name" in sus.columns:
        join_key = "Material_Name"
    else:
        raise ValueError("No common key to merge sustainability scores. Need Original_Index or Material_Name in BOTH files.")
    
    # keep only key + S columns
    sus_small = sus[[join_key] + S_cols].copy()
    
    # merge
    df = df.merge(sus_small, on=join_key, how="left")
    
    # rename the S columns to exactly S1..S10 (in case they were S1_something)
    rename_map = {S_cols[i-1]: f"S{i}" for i in range(1, 11)}
    df = df.rename(columns=rename_map)

    # --- pulizia DB elementare ---
    if "Z" not in db.columns:
        raise ValueError("Nel database elementi manca la colonna 'Z' (1..118).")

    db = db.dropna(subset=["Z"]).copy()
    db["Z"] = pd.to_numeric(db["Z"], errors="coerce")
    db = db.dropna(subset=["Z"]).copy()
    db["Z"] = db["Z"].astype(int)

    # --- estrazione simboli elementi (per proxy NON_MINED/REE) ---
    elem_col = None
    for c in db.columns:
        if "element" in str(c).lower():   # prende "Elements " o simili
            elem_col = c
            break

    elem_symbols_by_Z = None
    if elem_col is not None:
        tmp = db[["Z", elem_col]].dropna()
        elem_symbols_by_Z = {int(z): str(sym).strip() for z, sym in zip(tmp["Z"], tmp[elem_col])}

    # --- vettori proprietà: production + reserve (NaN se manca) ---
    v_prod_raw = _build_prop_vector(db, "production")
    v_res_raw  = _build_prop_vector(db, "reserve")

    # --- proxy fill (come tuo script) ---
    v_prod, v_res, proxy_info = _apply_proxies(v_prod_raw, v_res_raw, elem_symbols_by_Z)

    # --- build AF matrix (Materials x 118 elements) ---
    af_cols = [f"AF_{i}" for i in range(1, 119)]
    missing_af = [c for c in af_cols if c not in df.columns]
    if missing_af:
        raise ValueError(f"Mancano colonne AF nel file AF_vectors.csv (esempi): {missing_af[:5]}")

    af_matrix = df[af_cols].fillna(0.0).to_numpy(dtype=float)

    # --- weakest-link metrics (Pmax, Plong) ---
    df["Pmax_t_per_yr"] = _weakest_link_vectorized(af_matrix, v_prod)
    df["Plong_t"]       = _weakest_link_vectorized(af_matrix, v_res)

    # --- opzionale: metriche da DB (se esistono) ---
    try:
        v_hhi  = _build_prop_vector(db, "HHI")
        v_esg  = _build_prop_vector(db, "ESG")
        v_sr   = _build_prop_vector(db, "Supply risk")
        v_comp = _build_prop_vector(db, "Companionality (%)")

        df["HHI"] = _weighted_avg_with_nan_propagation(af_matrix, v_hhi)
        df["ESG"] = _weighted_avg_with_nan_propagation(af_matrix, v_esg)
        df["Supply risk"] = _weighted_avg_with_nan_propagation(af_matrix, v_sr)
        df["Companionality (%)"] = _weighted_avg_with_nan_propagation(af_matrix, v_comp)
    except Exception:
        pass

    # --- cleanup numerico ---
    for c in ["P1", "P2", "P3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # debug attrs
    df.attrs["proxy_info"] = proxy_info
    df.attrs["has_elem_symbols"] = bool(elem_symbols_by_Z)

    return df
        
    

# --- 3. MOTORE DI CALCOLO RANKING ---
def generate_linear_scores(n_tiers):
    return [round((i + 1) / n_tiers, 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, n_tiers, thresholds):
    scores = generate_linear_scores(n_tiers)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    for i in range(len(thresholds)):
        assigned[df[col_name] >= thresholds[i]] = scores[i+1]
    return assigned

# --- 4. INTERFACCIA APP ---
df = load_and_sync_data()

if df is None:
    st.error("Assicurati di avere 'AF_vectors.csv' e 'Materials Database 1.csv' nella cartella di lavoro.")
    st.stop()
w_ss = np.ones(10, dtype=float) / 10.0


# --- Compute SS from S1..S10 using user weights w_ss ---
S_cols = [f"S{i}" for i in range(1, 11)]
missing_S = [c for c in S_cols if c not in df.columns]
if missing_S:
    st.error(f"Missing sustainability columns in data: {missing_S}. SS cannot be computed.")
    st.stop()

S = df[S_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

# tua scelta: NaN -> 0.3
S = np.where(np.isnan(S), 0.3, S)
S = np.clip(S, 1e-12, 1.0)

# SS = exp( sum_i x_i log(S_i) )
df["SS"] = np.exp((np.log(S) * w_ss.reshape(1, -1)).sum(axis=1))




st.sidebar.markdown('<p class="settings-title">Settings</p>', unsafe_allow_html=True)
manual_thresholds = {"P1": [], "P2": [], "P3": []}
is_valid = True
w_ss = np.ones(10, dtype=float) / 10.0   # fallback (equal weights)
with st.sidebar:
    st.markdown('<p class="settings-title">Settings</p>', unsafe_allow_html=True)

    # =========================
    # 1) PERFORMANCE TIERS
    # =========================
    st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)

    # P1 Temperature
    sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2, key="sf_P1")
    sc_t = generate_linear_scores(sf_t)
    for i in range(sf_t - 1):
        val = st.number_input(
            f"Threshold for Score {sc_t[i+1]} (P1)",
            value=int(350 + (i * 50)),
            min_value=350,
            step=1,
            format="%d",
            key=f"p1_{i}"
        )
        manual_thresholds["P1"].append(float(val))

    # P2 Magnetization / P3 Coercivity
    for label, key, d_idx, d_val in [
        ("Magnetization (T)", "P2", 1, 0.4),
        ("Coercivity (T)", "P3", 3, 0.4),
    ]:
        st.markdown(f"**{label}**")
        sf = st.selectbox(f"Subcategories ({key})", [2, 3, 4, 5], index=d_idx, key=f"sf_{key}")
        sc = generate_linear_scores(sf)
        for i in range(sf - 1):
            v = st.number_input(
                f"Threshold for Score {sc[i+1]} ({key})",
                value=float(d_val + (i * 0.2)),
                min_value=float(d_val),
                key=f"t_{key}_{i}"
            )
            manual_thresholds[key].append(float(v))

        if key == "P2":
            sf_m = sf
        else:
            sf_c = sf

    # =========================
    # 2) PERFORMANCE WEIGHTS
    # =========================
    st.markdown('<div class="blue-section-header"><p>2. Performance Weights</p></div>', unsafe_allow_html=True)

    w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33, key="w_p1")
    rem = float(round(1.0 - w_p1, 2))
    w_p2 = st.slider("Weight P2 (Mag)", 0.0, rem, min(0.33, rem), key="w_p2")
    w_p3 = float(round(max(0.0, 1.0 - (w_p1 + w_p2)), 2))

    st.markdown(
        f"""
        <div class="custom-summary-box" style="padding:10px 12px; margin-top:10px;">
            <p style="margin:0; font-size:14px;"><b>Weight P3 (Coercivity)</b>: {w_p3:.2f}</p>
            <p style="margin:0; font-size:12px; opacity:0.8;">(auto = 1 − P1 − P2)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # 3) SCALABILITY VIEW
    # =========================
    st.markdown('<div class="blue-section-header"><p>3. Scalability View</p></div>', unsafe_allow_html=True)

    color_metric = st.selectbox(
        "Coloring Metric",
        ["SS", "HHI", "ESG", "Supply risk", "Companionality (%)"],
        index=0,
        key="color_metric"
    )

    # =========================
    # 3B) SUSTAINABILITY WEIGHTS
    # =========================
    st.markdown('<div class="blue-section-header"><p>3B. Sustainability Weights</p></div>', unsafe_allow_html=True)
    st.caption("SS = Π S_i^(x_i) with Σx_i = 1. Weights are auto-normalized.")

    default_w = [0.1] * 10
    w_in = []
    for i in range(1, 11):
        w_in.append(
            st.number_input(
                f"Weight x{i} for S{i}",
                min_value=0.0,
                max_value=1.0,
                value=default_w[i-1],
                step=0.01,
                key=f"w_s{i}"
            )
        )

    w_sum = float(np.sum(w_in))
    if w_sum <= 0:
        st.warning("All sustainability weights are 0. Using equal weights (0.1 each).")
        w_ss = np.ones(10, dtype=float) / 10.0
    else:
        w_ss = np.array(w_in, dtype=float) / w_sum

    st.markdown(
        f"""
        <div class="custom-summary-box" style="padding:10px 12px; margin-top:10px;">
            <p style="margin:0; font-size:14px;"><b>Σ weights</b>: {w_sum:.3f} → normalized to 1</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # 4) TOP-RIGHT TREND
    # =========================
    st.markdown('<div class="blue-section-header"><p>4. Top-right Trend</p></div>', unsafe_allow_html=True)

    trend_metrics = st.multiselect(
        "Metrics to test vs H = log(Pmax)+log(Plong)",
        ["SS", "HHI", "ESG", "Supply risk", "Companionality (%)"],
        default=["SS", "HHI", "ESG", "Supply risk", "Companionality (%)"],
        key="trend_metrics"
    )






# --- CALCOLI ---
p1_s = assign_tiered_scores(df, "P1", sf_t, manual_thresholds["P1"]) if "P1" in df.columns else 1.0
p2_s = assign_tiered_scores(df, "P2", sf_m, manual_thresholds["P2"]) if "P2" in df.columns else 1.0
p3_s = assign_tiered_scores(df, "P3", sf_c, manual_thresholds["P3"]) if "P3" in df.columns else 1.0

df["OPS"] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)

t1, t2, t3, t4 = st.tabs([
    "🏆 Pareto Ranking",
    "🏭 Scalability Map",
    "🔬 Stability Analysis",
    "📈 Top-right Trend"
])

with t1:
    colA, colB = st.columns([2, 1])
    pts = df[["OPS", "SS"]].to_numpy(dtype=float)

    efficient = np.ones(pts.shape[0], dtype=bool)
    for i, c in enumerate(pts):
        if efficient[i]:
            efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))

    df["Status"] = np.where(efficient, "Optimal Choice", "Standard")

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
        st.dataframe(
            df[efficient].sort_values(by="OPS", ascending=False)[show_cols],
            use_container_width=True,
            height=500
        )

with t2:
    st.markdown("### Scalability (Weakest-link)")
    st.caption("y = Pmax = min(P_i / x_i),  x = Plong = min(R_i / x_i).  (x_i = AF_i)")

    df_plot = df.dropna(subset=["Pmax_t_per_yr", "Plong_t"]).copy()

    metric_col = color_metric
    if metric_col not in df_plot.columns:
        st.warning(f"Colonna '{metric_col}' non trovata nel CSV. Uso 'SS' come fallback.")
        metric_col = "SS"

    if metric_col not in df_plot.columns:
        df_plot[metric_col] = np.nan

    short_label = "Comp. (%)" if metric_col == "Companionality (%)" else metric_col

    df_nonan = df_plot[df_plot[metric_col].notna()].copy()
    df_nan   = df_plot[df_plot[metric_col].isna()].copy()

    for d in (df_nonan, df_nan):
        d["Pmax_t_per_yr"] = pd.to_numeric(d["Pmax_t_per_yr"], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=1e-12)
        d["Plong_t"]       = pd.to_numeric(d["Plong_t"], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=1e-12)

    fig_sc = go.Figure()

    # --- NON-NaN: colorati + colorbar
    if len(df_nonan) > 0:
        fig_sc.add_trace(go.Scatter(
            x=df_nonan["Plong_t"].astype(float),
            y=df_nonan["Pmax_t_per_yr"].astype(float),
            mode="markers",
            name=short_label,
            marker=dict(
                size=np.where(df_nonan.get("Status", "Standard") == "Optimal Choice", 10, 7),
                color=pd.to_numeric(df_nonan[metric_col], errors="coerce"),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=short_label + " (grey = NaN)"),
                opacity=0.9,
            ),
            text=df_nonan["Material_Name"].astype(str) if "Material_Name" in df_nonan.columns else None,
            hovertemplate=(
                "%{text}<br>"
                "Plong=%{x:.3g}<br>"
                "Pmax=%{y:.3g}<br>"
                + metric_col + "=%{marker.color:.3g}<extra></extra>"
            ),
        ))

    # --- NaN: grigi (no colorbar)
    if len(df_nan) > 0:
        fig_sc.add_trace(go.Scatter(
            x=df_nan["Plong_t"].astype(float),
            y=df_nan["Pmax_t_per_yr"].astype(float),
            mode="markers",
            name=short_label + " (NaN)",
            marker=dict(
                size=np.where(df_nan.get("Status", "Standard") == "Optimal Choice", 10, 7),
                color="lightgrey",
                opacity=0.9,
            ),
            text=df_nan["Material_Name"].astype(str) if "Material_Name" in df_nan.columns else None,
            hovertemplate=(
                "%{text}<br>"
                "Plong=%{x:.3g}<br>"
                "Pmax=%{y:.3g}<br>"
                + metric_col + "=NaN<extra></extra>"
            ),
        ))

    fig_sc.update_layout(
    template="plotly_white",
    height=650,
    xaxis=dict(type="log", title="Long-term production (tons)  [min(R_i/x_i)]"),
    yaxis=dict(type="log", title="Max yearly production (t/yr) [min(P_i/x_i)]"),
    legend_title_text="Legend",
    legend=dict(orientation="h", y=-0.25, x=0.0),
)

    st.caption(f"Number of materials plotted: {len(df_plot)}")
    st.plotly_chart(fig_sc, use_container_width=True)
with t3:
    opts = df[df["Status"] == "Optimal Choice"]["Material_Name"].unique() if "Material_Name" in df.columns else []
    if len(opts) > 0:
        sel = st.selectbox("Select a Material to test:", opts)
        if st.button("Run Simulation ⚡"):
            idx = df[df["Material_Name"] == sel].index[0]
            rng = np.random.default_rng()
            W_sim = rng.dirichlet(np.array([w_p1, w_p2, w_p3]) * 50 + 1, 1000)
            s_vec = np.array([p1_s[idx], p2_s[idx], p3_s[idx]], dtype=float)
            c_ops = np.exp(np.dot(W_sim, np.log(s_vec + 1e-9)))
            fig_mc = px.scatter(
                x=c_ops,
                y=[df.loc[idx, "SS"]] * 1000,
                opacity=0.3,
            )
            fig_mc.update_layout(template="plotly_white")
            st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("No Pareto-optimal materials found with current settings.")



with t4:
    st.markdown("### Does a metric increase when moving top-right?")
    st.caption("We test each metric Y against H = log(Pmax) + log(Plong). NaN points are ignored.")

    base = df.copy()

    base = base.dropna(subset=["Pmax_t_per_yr", "Plong_t"]).copy()
    base = base[(base["Pmax_t_per_yr"] > 0) & (base["Plong_t"] > 0)].copy()

    if len(base) == 0:
        st.warning("No valid points with Pmax>0 and Plong>0.")
        st.stop()

    H = np.log(base["Pmax_t_per_yr"].to_numpy(dtype=float)) + np.log(base["Plong_t"].to_numpy(dtype=float))
    base["_H_"] = H

    label_map = {
        "Companionality (%)": "Comp. (%)",
        "Supply risk": "Supply risk",
        "HHI": "HHI",
        "ESG": "ESG",
        "SS": "SS",
    }

    cols = st.columns(2)
    col_idx = 0

    for metric in trend_metrics:
        if metric not in base.columns:
            st.info(f"Metric '{metric}' not found in dataframe → skipped.")
            continue

        tmp = base[["_H_", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric, "_H_"]).copy()

        if len(tmp) < 3:
            st.info(f"Not enough valid points for '{metric}' (n={len(tmp)}).")
            continue

        x = tmp["_H_"].to_numpy(dtype=float)
        y = tmp[metric].to_numpy(dtype=float)

        m, q = np.polyfit(x, y, 1)
        y_hat = m * x + q

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-30
        r2 = 1 - ss_res / ss_tot

        r = np.corrcoef(x, y)[0, 1]
        rho = pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson")

        dof = max(len(y) - 2, 1)
        sigma = np.std(y - y_hat) + 1e-12
        chi2 = np.sum(((y - y_hat) / sigma) ** 2)
        chi2_red = chi2 / dof

        short = label_map.get(metric, metric)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=short,
            marker=dict(size=8, opacity=0.85),
        ))

        xs = np.linspace(np.min(x), np.max(x), 200)
        fig.add_trace(go.Scatter(
            x=xs,
            y=m * xs + q,
            mode="lines",
            name="Trend",
        ))

        fig.update_layout(
            template="plotly_white",
            height=420,
            title=f"{short} vs H  (n={len(y)}, slope={m:.3g}, R²={r2:.3g})",
            xaxis_title="H = log(Pmax) + log(Plong)",
            yaxis_title=short,
            legend=dict(orientation="h", y=-0.25, x=0.0),
            margin=dict(l=40, r=20, t=60, b=60),
        )

        with cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True)
            st.write(
                f"**Stats** — n={len(y)} | slope={m:.4g} | intercept={q:.4g} | "
                f"R²={r2:.3f} | Pearson r={r:.3f} | Spearman ρ={rho:.3f} | χ²_red(proxy)={chi2_red:.2f}"
            )

        col_idx = 1 - col_idx
