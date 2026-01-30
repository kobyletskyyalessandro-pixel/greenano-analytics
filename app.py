import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    # reindex Z=1..118 (lascia NaN se manca)
    v = pd.Series(v.values, index=db["Z"].values).reindex(range(1, 119))
    return v.to_numpy(dtype=float)

def _weakest_link_vectorized(af_matrix: np.ndarray, v_elem: np.ndarray) -> np.ndarray:
    """
    af_matrix: (N,118) atomic fractions x_i
    v_elem: (118,) element property (production or reserves)
    return: (N,) min_i (v_i / x_i) for x_i>0
    """
    x = np.asarray(af_matrix, dtype=float)
    v = np.asarray(v_elem, dtype=float).reshape(1, -1)

    # ratio = v / x for x>0, else +inf (so it doesn't dominate min)
    ratio = np.where(x > 0, v / np.maximum(x, 1e-30), np.inf)

    # If any v is NaN where x>0 => ratio is NaN => min becomes NaN.
    # We handle NaN explicitly: treat NaN as inf so it won't break, but if a material uses an element with NaN
    # you actually want to see the issue. We'll choose: if any used element is NaN -> result NaN.
    used = (x > 0)
    has_nan_used = np.any(used & np.isnan(v), axis=1)
    out = np.min(ratio, axis=1)
    out = np.where(has_nan_used, np.nan, out)
    return out

def _apply_proxies(v_prod: np.ndarray, v_res: np.ndarray, elem_symbols_by_Z: dict | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fills missing values similarly to your script:
    - NON_MINED -> BIG
    - REE missing -> equal-share of remaining world totals
    - remaining NaN -> floor (min non-NaN)
    Returns v_prod_filled, v_res_filled, info dict
    """
    info = {"ree_missing_prod": [], "ree_missing_res": [], "note": ""}

    vp = v_prod.copy().astype(float)
    vr = v_res.copy().astype(float)

    # If no symbols, we can’t apply NON_MINED / REE logic properly
    if not elem_symbols_by_Z:
        # fallback: floor fill only
        floor_p = np.nanmin(vp)
        floor_r = np.nanmin(vr)
        vp = np.where(np.isnan(vp), floor_p, vp)
        vr = np.where(np.isnan(vr), floor_r, vr)
        info["note"] = "No element symbols found -> only floor proxy applied."
        return vp, vr, info

    # map arrays indices -> Z
    Zs = np.arange(1, 119)

    # NON_MINED -> BIG
    nm_mask = np.array([elem_symbols_by_Z.get(int(z), "").strip() in NON_MINED for z in Zs])
    vp[nm_mask] = np.where(np.isnan(vp[nm_mask]), BIG, vp[nm_mask])
    vr[nm_mask] = np.where(np.isnan(vr[nm_mask]), BIG, vr[nm_mask])

    # REE equal-share fill for missing (only if symbol in REE)
    ree_mask = np.array([elem_symbols_by_Z.get(int(z), "").strip() in REE for z in Zs])

    # production
    known_prod = np.nansum(vp[ree_mask])
    missing_prod_idx = np.where(ree_mask & np.isnan(vp))[0]
    rem_prod = max(WORLD_REO_PROD_2024 - known_prod, 0.0)
    if len(missing_prod_idx) > 0:
        fill_prod = rem_prod / len(missing_prod_idx)
        vp[missing_prod_idx] = fill_prod
        info["ree_missing_prod"] = [elem_symbols_by_Z.get(int(i+1), f"Z{i+1}") for i in missing_prod_idx]

    # reserves
    known_res = np.nansum(vr[ree_mask])
    missing_res_idx = np.where(ree_mask & np.isnan(vr))[0]
    rem_res = max(WORLD_REO_RESERVES - known_res, 0.0)
    if len(missing_res_idx) > 0:
        fill_res = rem_res / len(missing_res_idx)
        vr[missing_res_idx] = fill_res
        info["ree_missing_res"] = [elem_symbols_by_Z.get(int(i+1), f"Z{i+1}") for i in missing_res_idx]

    # floor fill remaining NaN
    floor_p = np.nanmin(vp)
    floor_r = np.nanmin(vr)
    vp = np.where(np.isnan(vp), floor_p, vp)
    vr = np.where(np.isnan(vr), floor_r, vr)

    return vp, vr, info

@st.cache_data
def load_and_sync_data():
    try:
        # Carica AF_vectors (Contiene P1, P2, P3 e AF_1..AF_118)
        df = pd.read_csv("AF_vectors.csv")

        # Carica Database Elementare
        db = pd.read_csv("Materials Database 1.csv")

        # Pulizia DB Elementare
        if "Z" not in db.columns:
            raise ValueError("Nel database elementi manca la colonna 'Z' (1..118).")
        db = db.dropna(subset=["Z"]).copy()
        db["Z"] = pd.to_numeric(db["Z"], errors="coerce").astype("Int64")
        db = db.dropna(subset=["Z"]).copy()
        db["Z"] = db["Z"].astype(int)

        # --- estrazione simboli elementi se presenti (per proxy NON_MINED/REE) ---
        elem_col = None
        # cerca una colonna che contenga "element" (robusto anche a "Elements ")
        for c in db.columns:
            if "element" in str(c).lower():
                elem_col = c
                break
        elem_symbols_by_Z = None
        if elem_col is not None:
            tmp = db[[ "Z", elem_col ]].dropna()
            elem_symbols_by_Z = {int(z): str(sym).strip() for z, sym in zip(tmp["Z"], tmp[elem_col])}

        # --- vettori proprietà (NaN se manca) ---
        v_prod_raw = _build_prop_vector(db, "production")
        v_res_raw  = _build_prop_vector(db, "reserve")

        # proxy fill (come tuo script)
        v_prod, v_res, proxy_info = _apply_proxies(v_prod_raw, v_res_raw, elem_symbols_by_Z)

        # Matrice frazioni atomiche (Materiali x 118 elementi)
        af_cols = [f"AF_{i}" for i in range(1, 119)]
        missing_af = [c for c in af_cols if c not in df.columns]
        if missing_af:
            raise ValueError(f"Mancano colonne AF nel file AF_vectors.csv (esempi): {missing_af[:5]}")

        af_matrix = df[af_cols].fillna(0.0).to_numpy(dtype=float)

        # --- Weakest-link metrics (come il tuo codice: min(P_i/x_i), min(R_i/x_i)) ---
        df["Pmax_t_per_yr"] = _weakest_link_vectorized(af_matrix, v_prod)
        df["Plong_t"]       = _weakest_link_vectorized(af_matrix, v_res)

        # Manteniamo anche il vecchio dot-product (non disturba), se ti serve in futuro
        df["Calc_Production"] = af_matrix @ np.nan_to_num(v_prod, nan=0.0)
        df["Calc_Reserves"]   = af_matrix @ np.nan_to_num(v_res,  nan=0.0)

        # OSS: se mancano S1..S10, creiamo uno sintetico (come AppNew originale)
        if not all(f"S{i}" in df.columns for i in range(1, 11)):
            # non abbiamo più v_risk/v_hhi/v_esg qui (perché tu hai chiesto SOLO scalability),
            # quindi usiamo un proxy semplice basato su Pmax/Plong (più alto = meglio) se serve.
            # Se invece hai già OSS nel csv, non lo tocchiamo.
            if "OSS" not in df.columns:
                # normalizza su [0,1] e media (evita log)
                def norm01(s):
                    s = pd.to_numeric(s, errors="coerce")
                    s = s.replace([np.inf, -np.inf], np.nan).fillna(np.nan)
                    mn, mx = np.nanmin(s), np.nanmax(s)
                    return (s - mn) / (mx - mn + 1e-12)

                a = norm01(np.log10(df["Pmax_t_per_yr"].replace(0, np.nan)))
                b = norm01(np.log10(df["Plong_t"].replace(0, np.nan)))
                df["OSS"] = (a + b) / 2.0

        for c in ["P1", "P2", "P3"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Salviamo info proxy come attributo “debug” (non rompe nulla)
        df.attrs["proxy_info"] = proxy_info
        df.attrs["has_elem_symbols"] = bool(elem_symbols_by_Z)

        return df

    except Exception as e:
        st.error(f"Errore caricamento o sincronizzazione database: {e}")
        return None

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

if df is not None:
    st.sidebar.markdown('<p class="settings-title">Settings</p>', unsafe_allow_html=True)
    manual_thresholds = {"P1": [], "P2": [], "P3": []}
    is_valid = True

    with st.sidebar:
        # SEZIONE 1: PERFORMANCE
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)

        # P1 Temperature
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(
                f"Threshold for Score {sc_t[i+1]} (P1)",
                value=int(350 + (i*50)),
                min_value=350,
                step=1,
                format="%d",
                key=f"p1_{i}"
            )
            manual_thresholds["P1"].append(float(val))

        # P2 Magnetization / P3 Coercivity
        for label, key, d_idx, d_val in [
            ("Magnetization (T)", "P2", 1, 0.4),
            ("Coercivity (T)", "P3", 3, 0.4)
        ]:
            st.markdown(f"**{label}**")
            sf = st.selectbox(f"Subcategories ({key})", [2, 3, 4, 5], index=d_idx, key=f"sf_{key}")
            sc = generate_linear_scores(sf)
            for i in range(sf - 1):
                v = st.number_input(
                    f"Threshold for Score {sc[i+1]} ({key})",
                    value=float(d_val + (i*0.2)),
                    min_value=float(d_val),
                    key=f"t_{key}_{i}"
                )
                manual_thresholds[key].append(float(v))
            if key == "P2":
                sf_m = sf
            else:
                sf_c = sf

        # SEZIONE 2: COEFFICIENTS
        st.markdown('<div class="blue-section-header"><p>2. Performance Weights</p></div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = round(1.0 - w_p1, 2)
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, rem, min(0.33, rem))
        w_p3 = round(max(0.0, 1.0 - (w_p1 + w_p2)), 2)

        # ✅ (MODIFICA 1) Mostra sempre il peso P3 aggiornato
        st.markdown(
            f"""
            <div class="custom-summary-box" style="padding:10px 12px; margin-top:10px;">
                <p style="margin:0; font-size:14px;"><b>Weight P3 (Coercivity)</b>: {w_p3:.2f}</p>
                <p style="margin:0; font-size:12px; opacity:0.8;">(auto = 1 − P1 − P2)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # SEZIONE 3: VIEW SETTINGS (SCALABILITY)
        st.markdown('<div class="blue-section-header"><p>3. Scalability View</p></div>', unsafe_allow_html=True)
        color_metric = st.selectbox("Coloring Metric", ["OSS"])

    # --- CALCOLI ---
    if is_valid:
        p1_s = assign_tiered_scores(df, "P1", sf_t, manual_thresholds["P1"]) if "P1" in df.columns else 1.0
        p2_s = assign_tiered_scores(df, "P2", sf_m, manual_thresholds["P2"]) if "P2" in df.columns else 1.0
        p3_s = assign_tiered_scores(df, "P3", sf_c, manual_thresholds["P3"]) if "P3" in df.columns else 1.0

        df["OPS"] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)

        # --- TABS ---
        t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "🔬 Stability Analysis"])

        with t1:
            colA, colB = st.columns([2, 1])
            pts = df[["OPS", "OSS"]].to_numpy(dtype=float)
            efficient = np.ones(pts.shape[0], dtype=bool)
            for i, c in enumerate(pts):
                if efficient[i]:
                    efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))

            df["Status"] = np.where(efficient, "Optimal Choice", "Standard")

            with colA:
                fig = px.scatter(
                    df, x="OPS", y="OSS", color="Status",
                    hover_name="Material_Name" if "Material_Name" in df.columns else None,
                    color_discrete_map={"Optimal Choice": "#1e3a8a", "Standard": "#cbd5e1"}
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                st.markdown("**Top Pareto Materials**")
                show_cols = [c for c in ["Material_Name", "OPS", "OSS"] if c in df.columns]
                st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[show_cols],
                             use_container_width=True, height=500)

        with t2:
            # ✅ (MODIFICA 2) Scalability weakest-link: Plong vs Pmax, log-log, colored by OSS
            st.markdown("### Scalability (Weakest-link)")
            st.caption("y = Pmax = min(P_i / x_i),  x = Plong = min(R_i / x_i).  (x_i = AF_i)")

            if "Pmax_t_per_yr" not in df.columns or "Plong_t" not in df.columns:
                st.error("Mancano Pmax_t_per_yr / Plong_t. Controlla che AF_1..AF_118 e production/reserve siano disponibili.")
            else:
                df_plot = df.dropna(subset=["Pmax_t_per_yr", "Plong_t", "OSS"]).copy()

                # clip for log-scale safety
                df_plot["Pmax_t_per_yr"] = pd.to_numeric(df_plot["Pmax_t_per_yr"], errors="coerce").clip(lower=1e-12)
                df_plot["Plong_t"] = pd.to_numeric(df_plot["Plong_t"], errors="coerce").clip(lower=1e-12)

                fig_sc = px.scatter(
                df_plot,
                x="Plong_t",
                y="Pmax_t_per_yr",
                color="OSS",
                size=np.where(df_plot.get("Status", "Standard") == "Optimal Choice", 5, 3),
                    size_max=6,
                    hover_name="Material_Name" if "Material_Name" in df_plot.columns else None,
                    log_x=True,
                    log_y=True,
                    labels={
                        "Plong_t": "Long-term production (tons)  [min(R_i/x_i)]",
                        "Pmax_t_per_yr": "Max yearly production (t/yr) [min(P_i/x_i)]",
                        "OSS": "OSS"
                    },
                    color_continuous_scale="Viridis",
                )
                fig_sc.update_layout(template="plotly_white", height=650)
                st.plotly_chart(fig_sc, use_container_width=True)

                # Numerazione punti (se c'è Original_Index)
                if "Original_Index" in df_plot.columns:
                    st.caption("Point numbering available in hover via Original_Index.")
                    # aggiungi Original_Index in hover
                    fig_sc.update_traces(customdata=df_plot[["Original_Index"]].to_numpy())
                    fig_sc.update_traces(hovertemplate=(
                        "%{hovertext}<br>" +
                        "Plong=%{x:.3g}<br>Pmax=%{y:.3g}<br>" +
                        "OSS=%{marker.color:.3f}<br>" +
                        "Original_Index=%{customdata[0]}<extra></extra>"
                    ))

                # Debug proxy info (collassabile)
                with st.expander("Proxy debug (REE / NON_MINED / floors)"):
                    info = df.attrs.get("proxy_info", {})
                    st.write(info)
                    st.write("Has element symbols:", df.attrs.get("has_elem_symbols", False))

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
                        y=[df.loc[idx, "OSS"]] * 1000,
                        opacity=0.3,
                        color_discrete_sequence=["#1e3a8a"]
                    )
                    fig_mc.update_layout(template="plotly_white")
                    st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.info("No Pareto-optimal materials found with current settings.")

else:
    st.error("Assicurati di avere 'AF_vectors.csv' e 'Materials Database 1.csv' nella cartella di lavoro.")
