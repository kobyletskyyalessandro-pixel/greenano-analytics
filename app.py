import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from figures_paper_engine import preference_sweep_ops, threshold_sensitivity_simple

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="GreeNano Analytics", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# HELPERS
# ----------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^-0-9.]", "", regex=True), errors="coerce").fillna(0.0)

def assign_tiered_scores(df: pd.DataFrame, col: str, score_levels: list[float], thresholds: list[float]) -> np.ndarray:
    """
    thresholds: list of 4 values -> 5 tiers
    score_levels: list of 5 values
    """
    x = df[col].to_numpy(dtype=float)
    thr = sorted(list(thresholds))
    levels = list(score_levels)
    out = np.empty_like(x, dtype=float)
    for i, v in enumerate(x):
        idx = 0
        while idx < len(thr) and v > thr[idx]:
            idx += 1
        idx = min(idx, len(levels) - 1)
        out[i] = levels[idx]
    return out

@st.cache_data
def load_and_sync_data() -> pd.DataFrame:
    # --- required files in repo root:
    # AF_vectors.csv
    # Materials Database 1.csv
    af = pd.read_csv("AF_vectors.csv")
    db = pd.read_csv("Materials Database 1.csv")

    # cleanup
    for c in ["P1", "P2", "P3"]:
        if c in af.columns:
            af[c] = clean_numeric(af[c])

    # sync by Material_Name if possible, else just keep af
    if "Material_Name" in af.columns and "Material_Name" in db.columns:
        df = af.merge(db, on="Material_Name", how="left")
    else:
        df = af.copy()

    # Ensure Material_Name exists
    if "Material_Name" not in df.columns:
        df["Material_Name"] = df.index.astype(str)

    # Calculate scalability proxies from AF vectors (if present)
    af_cols = [c for c in df.columns if c.startswith("AF_")]
    if len(af_cols) > 0:
        # placeholder: if your db already has element vectors, replace here
        # For now keep compatibility: create non-null columns used in the UI
        if "Calc_Production" not in df.columns:
            df["Calc_Production"] = 1.0
        if "Calc_Reserves" not in df.columns:
            df["Calc_Reserves"] = 1.0
        if "Calc_Supply_Risk" not in df.columns:
            df["Calc_Supply_Risk"] = 0.0
        if "Calc_HHI" not in df.columns:
            df["Calc_HHI"] = 0.0
        if "Calc_ESG" not in df.columns:
            df["Calc_ESG"] = 0.0

    # OSS fallback if not present
    if "OSS" not in df.columns:
        # if S1..S10 exist, average them, otherwise proxy from risk/hhi/esg
        s_cols = [f"S{i}" for i in range(1, 11) if f"S{i}" in df.columns]
        if len(s_cols) == 10:
            df["OSS"] = df[s_cols].apply(clean_numeric).mean(axis=1)
        else:
            df["OSS"] = (1.0 / (1.0 + df.get("Calc_Supply_Risk", 0.0))).astype(float)

    return df

# ----------------------------
# APP
# ----------------------------
st.title("GreeNano Analytics")

try:
    df = load_and_sync_data()
    is_valid = True
except Exception as e:
    is_valid = False
    st.error(f"Errore caricamento dati: {e}")

if not is_valid:
    st.error("Assicurati di avere 'AF_vectors.csv' e 'Materials Database 1.csv' nella cartella root della repo.")
    st.stop()

# Sidebar settings
st.sidebar.markdown("## Settings")

# Score tiers (5 levels)
score_levels = [0.2, 0.4, 0.6, 0.8, 1.0]

st.sidebar.markdown("### OPS weights")
w_p1 = st.sidebar.slider("Weight P1", 0.0, 1.0, 0.33, 0.01)
w_p2 = st.sidebar.slider("Weight P2", 0.0, 1.0, 0.33, 0.01)
w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
st.sidebar.caption(f"Weight P3 auto = {w_p3:.2f} (somma = 1)")

st.sidebar.markdown("### Manual thresholds (4 cutoffs)")
thr_p1 = [
    st.sidebar.number_input("P1 thr1", value=0.2),
    st.sidebar.number_input("P1 thr2", value=0.4),
    st.sidebar.number_input("P1 thr3", value=0.6),
    st.sidebar.number_input("P1 thr4", value=0.8),
]
thr_p2 = [
    st.sidebar.number_input("P2 thr1", value=0.2),
    st.sidebar.number_input("P2 thr2", value=0.4),
    st.sidebar.number_input("P2 thr3", value=0.6),
    st.sidebar.number_input("P2 thr4", value=0.8),
]
thr_p3 = [
    st.sidebar.number_input("P3 thr1", value=0.2),
    st.sidebar.number_input("P3 thr2", value=0.4),
    st.sidebar.number_input("P3 thr3", value=0.6),
    st.sidebar.number_input("P3 thr4", value=0.8),
]

color_metric = st.sidebar.selectbox("Coloring Metric (Scalability Map)", ["OSS", "Calc_Supply_Risk", "Calc_HHI", "Calc_ESG"])

# Calculations
p1_s = assign_tiered_scores(df, "P1", score_levels, thr_p1)
p2_s = assign_tiered_scores(df, "P2", score_levels, thr_p2)
p3_s = assign_tiered_scores(df, "P3", score_levels, thr_p3)

df["P1_score"] = p1_s
df["P2_score"] = p2_s
df["P3_score"] = p3_s

# OPS (geometric product of tier scores)
df["OPS"] = (p1_s ** w_p1) * (p2_s ** w_p2) * (p3_s ** w_p3)

# Pareto flag
pts = df[["OPS", "OSS"]].to_numpy(dtype=float)
efficient = np.ones(pts.shape[0], dtype=bool)
for i, c in enumerate(pts):
    if efficient[i]:
        efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))
df["Status"] = np.where(efficient, "Optimal Choice", "Standard")

# Tabs
t1, t2, t3, t4 = st.tabs([" Pareto Ranking", " Scalability Map", " Stability Analysis", " Paper Figures"])

with t1:
    colA, colB = st.columns([2, 1])

    with colA:
        fig = px.scatter(
            df,
            x="OPS",
            y="OSS",
            color="Status",
            hover_name="Material_Name",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("**Top Pareto Materials**")
        st.dataframe(
            df[efficient].sort_values(by="OPS", ascending=False)[["Material_Name", "OPS", "OSS"]],
            use_container_width=True,
            height=500,
        )

with t2:
    st.markdown("### Resource Scalability (Calculated / Proxy)")
    df_plot = df.copy()
    df_plot["Calc_Production"] = pd.to_numeric(df_plot["Calc_Production"], errors="coerce").fillna(0).clip(lower=1e-1)
    df_plot["Calc_Reserves"] = pd.to_numeric(df_plot["Calc_Reserves"], errors="coerce").fillna(0).clip(lower=1e-1)

    fig_sc = px.scatter(
        df_plot,
        x="Calc_Reserves",
        y="Calc_Production",
        color=color_metric,
        size=np.where(efficient, 15, 8),
        symbol=np.where(efficient, "star", "circle"),
        hover_name="Material_Name",
        hover_data=["Calc_Supply_Risk", "Calc_HHI", "Calc_ESG"],
        log_x=True,
        log_y=True,
        color_continuous_scale="Viridis",
        labels={"Calc_Reserves": "Calculated Global Reserves (t)", "Calc_Production": "Calculated Production (t/yr)"},
    )
    fig_sc.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.info("Stars represent materials currently on the Pareto Frontier.")

with t3:
    opts = df[efficient]["Material_Name"].unique()
    if len(opts) == 0:
        st.warning("Nessun materiale sul Pareto Front (controlla soglie/pesi).")
    else:
        sel = st.selectbox("Select a Material to test:", opts)
        n_sims = st.slider("Simulations", 200, 5000, 1000, 200)
        conc = st.slider("Dirichlet concentration (higher = more stable weights)", 5.0, 200.0, 50.0, 5.0)

        if st.button("Run Simulation"):
            idx = df.index[df["Material_Name"] == sel][0]
            # build small df for engine: just one row, but we show distribution of OPS when weights vary
            one = df.loc[[idx], ["Material_Name", "P1_score", "P2_score", "P3_score"]].copy()
            res = preference_sweep_ops(one, ["P1_score", "P2_score", "P3_score"], np.array([w_p1, w_p2, w_p3]), n_sims=n_sims, concentration=conc, seed=0)
            st.dataframe(res, use_container_width=True)

            # quick plot: OPS distribution simulation (we emulate by resampling weights and computing OPS directly)
            rng = np.random.default_rng(0)
            alpha = conc * np.array([w_p1, w_p2, w_p3]) + 1.0
            W = rng.dirichlet(alpha, size=n_sims)
            s_vec = np.array([df.loc[idx, "P1_score"], df.loc[idx, "P2_score"], df.loc[idx, "P3_score"]], dtype=float)
            ops_sim = np.exp(W @ np.log(np.clip(s_vec, 1e-12, None)))
            fig_mc = px.histogram(x=ops_sim, nbins=40)
            fig_mc.update_layout(template="plotly_white", title="OPS distribution under weight uncertainty")
            st.plotly_chart(fig_mc, use_container_width=True)

with t4:
    st.markdown("### Paper Figures (FiguresPaper port)")
    st.caption("Analisi di stabilità e sensibilità: output pensato per figure riproducibili.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Case 1: Preference sweep (Dirichlet weights)")
        n_sims = st.slider("Case1 sims", 500, 20000, 3000, 500)
        conc = st.slider("Case1 concentration", 5.0, 200.0, 50.0, 5.0)
        topk = st.slider("Show top-K by top1_freq", 5, 50, 15, 1)

        if st.button("Run Case 1"):
            res = preference_sweep_ops(
                df[["Material_Name", "P1_score", "P2_score", "P3_score"]],
                ["P1_score", "P2_score", "P3_score"],
                np.array([w_p1, w_p2, w_p3]),
                n_sims=n_sims,
                concentration=conc,
                seed=0,
            )
            st.dataframe(res.head(topk), use_container_width=True)

            fig = px.bar(res.head(topk), x="Material_Name", y="top1_freq")
            fig.update_layout(template="plotly_white", title="Top-1 frequency under weight uncertainty")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Case 2: Threshold sensitivity (simple)")
        st.caption("Misura Spearman vs baseline cambiando le soglie di discretizzazione (qui su P1).")

        # generate a few threshold variants around current
        base = thr_p1
        variants = [
            base,
            [base[0]*0.9, base[1]*0.9, base[2]*0.9, base[3]*0.9],
            [base[0]*1.1, base[1]*1.1, base[2]*1.1, base[3]*1.1],
            [base[0]*0.95, base[1]*1.05, base[2]*0.95, base[3]*1.05],
        ]

        if st.button("Run Case 2"):
            out = threshold_sensitivity_simple(df, "P1", variants, [0.2, 0.4, 0.6, 0.8, 1.0])
            st.dataframe(out, use_container_width=True)

            fig2 = px.bar(out, x="thresholds", y="spearman_vs_base")
            fig2.update_layout(template="plotly_white", title="Threshold sensitivity (Spearman vs base)")
            st.plotly_chart(fig2, use_container_width=True)
