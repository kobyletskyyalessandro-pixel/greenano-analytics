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

# --- 2. MOTORE DI CARICAMENTO E PULIZIA ---

def clean_numeric(series):
    """Pulisce stringhe con virgole e testi e converte in numeri."""
    return pd.to_numeric(series.astype(str).str.replace(r'[^-0-9.]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def load_and_sync_data():
    try:
        df = pd.read_csv("AF_vectors.csv")
        db = pd.read_csv("Materials Database 1.csv")
        db = db.dropna(subset=['Z'])
        db['Z'] = db['Z'].astype(int)
        
        def get_prop_vector(col_keyword):
            col_name = [c for c in db.columns if col_keyword.lower() in c.lower()][0]
            # Usiamo una conversione sicura elemento per elemento
            return db.set_index('Z')[col_name].apply(lambda x: clean_numeric(pd.Series([x])).iloc[0]).reindex(range(1, 119)).fillna(0).values

        v_prod = get_prop_vector('production')
        v_res = get_prop_vector('reserve')
        v_risk = get_prop_vector('risk')
        v_hhi = get_prop_vector('HHI')
        v_esg = get_prop_vector('ESG')

        af_cols = [f'AF_{i}' for i in range(1, 119)]
        af_matrix = df[af_cols].fillna(0).values

        df['Calc_Production'] = af_matrix @ v_prod
        df['Calc_Reserves'] = af_matrix @ v_res
        df['Calc_Supply_Risk'] = af_matrix @ v_risk
        df['Calc_HHI'] = af_matrix @ v_hhi
        df['Calc_ESG'] = af_matrix @ v_esg

        # Fallback per OSS se mancano colonne S1-S10
        if not all(f'S{i}' in df.columns for i in range(1, 11)):
            def norm(s): return 1 - (s - s.min()) / (s.max() - s.min() + 1e-9)
            df['OSS'] = (norm(df['Calc_Supply_Risk']) + norm(df['Calc_HHI']) + norm(df['Calc_ESG'])) / 3
            
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Errore caricamento o sincronizzazione database: {e}")
        return None

# --- 3. LOGICA CALCOLO RANKING ---

def generate_linear_scores(n_tiers):
    return [round((i + 1) / n_tiers, 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, n_tiers, thresholds):
    scores = generate_linear_scores(n_tiers)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    for i in range(len(thresholds)):
        assigned[df[col_name] >= thresholds[i]] = scores[i+1]
    return assigned

# --- 4. INTERFACCIA ---
df = load_and_sync_data()

if df is not None:
    st.sidebar.markdown('<p class="settings-title">Control Panel</p>', unsafe_allow_html=True)
    manual_thresholds = {'P1': [], 'P2': [], 'P3': []}
    is_valid = True
    
    with st.sidebar:
        # SEZIONE 1: PERFORMANCE
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        # P1 Temperature
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Threshold for Score {sc_t[i+1]} (P1)", value=int(350 + (i*50)), min_value=350, step=1, format="%d", key=f"p1_{i}")
            manual_thresholds['P1'].append(float(val))
        
        # P2/P3
        for label, key, d_idx, d_val in [("Magnetization (T)", "P2", 1, 0.4), ("Coercivity (T)", "P3", 3, 0.4)]:
            st.markdown(f"**{label}**")
            sf = st.selectbox(f"Subcategories ({key})", [2, 3, 4, 5], index=d_idx, key=f"sf_{key}")
            sc = generate_linear_scores(sf)
            for i in range(sf - 1):
                v = st.number_input(f"Threshold for Score {sc[i+1]} ({key})", value=d_val+(i*0.2), min_value=d_val, key=f"t_{key}_{i}")
                manual_thresholds[key].append(v)
            if key == "P2": sf_m = sf
            else: sf_c = sf

        # SEZIONE 3: VIEW SETTINGS
        st.markdown('<div class="blue-section-header"><p>2. Scalability Settings</p></div>', unsafe_allow_html=True)
        use_log = st.checkbox("Use Logarithmic Scale", value=False)
        color_metric = st.selectbox("Coloring Metric", ["OSS", "Calc_Supply_Risk", "Calc_HHI", "Calc_ESG"])
        point_size_val = st.slider("Point Size", 5, 20, 8)

    # --- CALCOLI ---
    p1_s = assign_tiered_scores(df, 'P1', sf_t, manual_thresholds['P1'])
    p2_s = assign_tiered_scores(df, 'P2', sf_m, manual_thresholds['P2'])
    p3_s = assign_tiered_scores(df, 'P3', sf_c, manual_thresholds['P3'])
    
    w_p1, w_p2, w_p3 = 0.33, 0.33, 0.34 # Default bilanciato o usa slider se preferisci
    df['OPS'] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)
    
    # --- TABS ---
    t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "🔬 Stability Analysis"])

    with t1:
        colA, colB = st.columns([2, 1])
        pts = df[['OPS', 'OSS']].to_numpy()
        efficient = np.ones(pts.shape[0], dtype=bool)
        for i, c in enumerate(pts):
            if efficient[i]: efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))
        df['Status'] = np.where(efficient, 'Optimal Choice', 'Standard')
        
        with colA:
            fig = px.scatter(df, x='OPS', y='OSS', color='Status', hover_name='Material_Name',
                             color_discrete_map={'Optimal Choice': '#1e3a8a', 'Standard': '#cbd5e1'})
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.markdown("**Top Pareto Materials**")
            st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], use_container_width=True, height=500)

    with t2:
        st.markdown("### Resource Scalability (Vector Product)")
        
        df_plot = df.copy()
        if use_log:
            df_plot['Calc_Production'] = df_plot['Calc_Production'].clip(lower=1e-1)
            df_plot['Calc_Reserves'] = df_plot['Calc_Reserves'].clip(lower=1e-1)
        
        fig_sc = px.scatter(df_plot, x='Calc_Reserves', y='Calc_Production', 
                            color=color_metric, 
                            hover_name='Material_Name', # MOSTRA IL NOME SUL PUNTATORE
                            hover_data=['Chemical_Formula', 'Calc_Supply_Risk', 'Calc_HHI', 'Calc_ESG'],
                            log_x=use_log, log_y=use_log,
                            color_continuous_scale="Viridis",
                            labels={'Calc_Reserves': 'Global Reserves (t)', 'Calc_Production': 'Annual Production (t/yr)'})
        
        # PUNTI PIÙ PICCOLI E PULITI
        fig_sc.update_traces(marker=dict(size=point_size_val, line=dict(width=0.5, color='white')))
        fig_sc.update_layout(template="plotly_white", height=650)
        st.plotly_chart(fig_sc, use_container_width=True)

    with t3:
        opts = df[efficient]['Material_Name'].unique()
        if len(opts) > 0:
            sel = st.selectbox("Select material:", opts)
            if st.button("Run Simulation ⚡"):
                idx = df[df['Material_Name'] == sel].index[0]
                rng = np.random.default_rng()
                W_sim = rng.dirichlet(np.array([0.33, 0.33, 0.34])*50 + 1, 1000)
                s_vec = np.array([p1_s[idx], p2_s[idx], p3_s[idx]])
                c_ops = np.exp(np.dot(W_sim, np.log(s_vec + 1e-9)))
                fig_mc = px.scatter(x=c_ops, y=[df.loc[idx, 'OSS']]*1000, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                st.plotly_chart(fig_mc, use_container_width=True)
else:
    st.error("Upload 'AF_vectors.csv' and 'Materials Database 1.csv' to the directory.")
