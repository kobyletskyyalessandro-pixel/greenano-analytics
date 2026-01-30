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
    return pd.to_numeric(series.astype(str).str.replace(r'[^-0.9.]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def load_and_sync_data():
    try:
        # Caricamento file originali
        df = pd.read_csv("AF_vectors.csv")
        db = pd.read_csv("Materials Database 1.csv")
        
        # Pulizia Database Elementare
        db = db.dropna(subset=['Z'])
        db['Z'] = db['Z'].astype(int)
        
        # Mappatura Proprietà Elementari (Z -> Valore)
        def get_prop_map(col_keyword):
            col_name = [c for c in db.columns if col_keyword.lower() in c.lower()][0]
            return db.set_index('Z')[col_name].apply(lambda x: clean_numeric(pd.Series([x])).iloc[0]).to_dict()

        map_prod = get_prop_map('production')
        map_res = get_prop_map('reserve')
        map_risk = get_prop_map('risk')
        map_hhi = get_prop_map('HHI')
        map_esg = get_prop_map('ESG')

        # Liste per il calcolo finale
        af_cols = [f'AF_{i}' for i in range(1, 119)]
        af_matrix = df[af_cols].fillna(0).values
        
        # Elementi non limitanti (H, C, N, O, F, Cl, gas nobili) - dal Notebook
        NON_LIMITING = {1, 6, 7, 8, 9, 10, 17, 18, 36, 54}
        BIG_VAL = 1e30

        # --- CALCOLO WEAKEST LINK (Pmax e Rmax) ---
        p_max_list, r_max_list = [], []
        
        for i in range(len(df)):
            row_af = af_matrix[i]
            pot_p, pot_r = [], []
            
            for z_idx, x_i in enumerate(row_af):
                z = z_idx + 1 # Z parte da 1
                if x_i > 0:
                    if z in NON_LIMITING:
                        continue
                    # Potenziale = Offerta Mondiale / Frazione Atomica
                    pot_p.append(map_prod.get(z, BIG_VAL) / (x_i + 1e-12))
                    pot_r.append(map_res.get(z, BIG_VAL) / (x_i + 1e-12))
            
            p_max_list.append(min(pot_p) if pot_p else BIG_VAL)
            r_max_list.append(min(pot_r) if pot_r else BIG_VAL)

        df['Pmax_Bottleneck'] = p_max_list
        df['Rmax_Bottleneck'] = r_max_list

        # --- CALCOLO PRODOTTO VETTORIALE (Per indici di rischio) ---
        v_risk = np.array([map_risk.get(z, 0) for z in range(1, 119)])
        v_hhi = np.array([map_hhi.get(z, 0) for z in range(1, 119)])
        v_esg = np.array([map_esg.get(z, 0) for z in range(1, 119)])

        df['Calc_Supply_Risk'] = af_matrix @ v_risk
        df['Calc_HHI'] = af_matrix @ v_hhi
        df['Calc_ESG'] = af_matrix @ v_esg

        # Sostenibilità (OSS) di fallback
        if not all(f'S{i}' in df.columns for i in range(1, 11)):
            def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
            df['OSS'] = 1 - (norm(df['Calc_Supply_Risk']) + norm(df['Calc_HHI'])) / 2
            
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Errore nella sincronizzazione dei dati: {e}")
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
    
    with st.sidebar:
        # SEZIONE 1: PERFORMANCE
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Threshold Score {sc_t[i+1]} (P1)", value=int(350 + (i*50)), min_value=350, step=1, format="%d", key=f"p1_{i}")
            manual_thresholds['P1'].append(float(val))
        
        for label, key, d_idx, d_val in [("Magnetization (T)", "P2", 1, 0.4), ("Coercivity (T)", "P3", 3, 0.4)]:
            st.markdown(f"**{label}**")
            sf = st.selectbox(f"Subcategories ({key})", [2, 3, 4, 5], index=d_idx, key=f"sf_{key}")
            sc = generate_linear_scores(sf)
            for i in range(sf - 1):
                v = st.number_input(f"Threshold Score {sc[i+1]} ({key})", value=d_val+(i*0.2), min_value=d_val, key=f"t_{key}_{i}")
                manual_thresholds[key].append(v)
            if key == "P2": sf_m = sf
            else: sf_c = sf

        # SEZIONE 2: SCALABILITY SETTINGS
        st.markdown('<div class="blue-section-header"><p>2. Scalability Settings</p></div>', unsafe_allow_html=True)
        outlier_percentile = st.slider("Exclude Top % Abundant (Cloud Zoom)", 0, 10, 2)
        color_metric = st.selectbox("Coloring Metric", ["OSS", "Calc_Supply_Risk", "Calc_HHI", "Calc_ESG"])
        point_size_val = st.slider("Point Size", 2, 10, 4)
        point_opacity = st.slider("Point Opacity", 0.1, 1.0, 0.6)

    # Calcoli Ranking
    p1_s = assign_tiered_scores(df, 'P1', sf_t, manual_thresholds['P1'])
    p2_s = assign_tiered_scores(df, 'P2', sf_m, manual_thresholds['P2'])
    p3_s = assign_tiered_scores(df, 'P3', sf_c, manual_thresholds['P3'])
    df['OPS'] = np.power(p1_s, 0.33) * np.power(p2_s, 0.33) * np.power(p3_s, 0.34)
    
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
        st.markdown("### Max Production Capacity (Bottleneck Analysis)")
        st.caption("Scalability is determined by the rarest element in the formula (Weakest Link logic).")

        # Filtro outlier per evitare lo schiacciamento
        cutoff_p = np.percentile(df['Pmax_Bottleneck'], 100 - outlier_percentile)
        cutoff_r = np.percentile(df['Rmax_Bottleneck'], 100 - outlier_percentile)
        df_plot = df[(df['Pmax_Bottleneck'] <= cutoff_p) & (df['Rmax_Bottleneck'] <= cutoff_r)].copy()
        
        fig_sc = px.scatter(df_plot, 
                            x='Rmax_Bottleneck', 
                            y='Pmax_Bottleneck', 
                            color=color_metric, 
                            hover_name='Material_Name',
                            hover_data={
                                'Chemical_Formula': True,
                                'Rmax_Bottleneck': ':.2e',
                                'Pmax_Bottleneck': ':.2e',
                                'Calc_Supply_Risk': ':.2f'
                            },
                            color_continuous_scale="Viridis",
                            labels={'Rmax_Bottleneck': 'Global Reserves Capacity (t)', 'Pmax_Bottleneck': 'Annual Production Capacity (t/yr)'})
        
        fig_sc.update_traces(marker=dict(size=point_size_val, opacity=point_opacity, line=dict(width=0)))
        fig_sc.update_layout(template="plotly_white", height=700)
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
