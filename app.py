import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="GreeNano Platform", layout="wide")

st.markdown("""
    <style>
    :root { --primary: #1e3a8a; }
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .blue-header { background-color: #1e3a8a; padding: 10px; border-radius: 8px; margin: 15px 0 10px 0; color: white; font-weight: 700; }
    div[data-baseweb="input"], div[data-baseweb="select"] > div { background-color: white !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important; }
    input, span { color: #1e3a8a !important; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGICA CARICAMENTO ---
def force_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^-0.9.]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def load_and_sync_data():
    try:
        # Carica AF_vectors
        df = pd.read_csv("AF_vectors.csv")
        
        # Carica Database Elementare (gestendo potenziali righe di intestazione sporche)
        db = pd.read_csv("Materials Database 1.csv", header=0)
        
        # Pulizia colonna Z
        z_col = [c for c in db.columns if c.strip().upper() == 'Z'][0]
        db['Z_idx'] = force_numeric(db[z_col]).astype(int)
        db = db[db['Z_idx'].between(1, 118)]

        # Trova colonne chiave per scalabilità
        prod_col = [c for c in db.columns if 'production' in c.lower() and 'year' in c.lower()][0]
        res_col = [c for c in db.columns if 'reserve' in c.lower() and 'tons' in c.lower()][0]
        risk_col = [c for c in db.columns if 'risk' in c.lower()][0]
        hhi_col = [c for c in db.columns if 'HHI' in c.upper()][0]
        
        # Estrazione vettori (Z 1-118)
        def get_v(col):
            return db.set_index('Z_idx')[col].apply(lambda x: force_numeric(pd.Series([x])).iloc[0]).reindex(range(1, 119)).fillna(0).values

        v_p = get_v(prod_col)
        v_r = get_v(res_col)
        v_risk = get_v(risk_col)
        v_hhi = get_v(hhi_col)

        # Matrice Frazioni Atomiche (AF_1...AF_118)
        af_cols = [f'AF_{i}' for i in range(1, 119)]
        af_matrix = df[af_cols].fillna(0).values

        # PRODOTTO VETTORIALE
        df['Calc_Prod'] = af_matrix @ v_p
        df['Calc_Res'] = af_matrix @ v_r
        df['Calc_Risk'] = af_matrix @ v_risk
        df['Calc_HHI'] = af_matrix @ v_hhi

        return df
    except Exception as e:
        st.error(f"Errore tecnico: {e}. Verifica i nomi delle colonne nel database.")
        return None

# --- 3. UI ---
df = load_sync_data()

if df is not None:
    st.sidebar.subheader("Settings")
    manual_t = {'P1': [], 'P2': [], 'P3': []}
    
    with st.sidebar:
        st.markdown('<div class="blue-header">1. Performance Tiers</div>', unsafe_allow_html=True)
        # P1 Temperature
        sf_t = st.selectbox("Tiers (P1)", [2, 3, 4, 5], index=2)
        scores = [round((i+1)/sf_t, 2) for i in range(sf_t)]
        for i in range(sf_t - 1):
            val = st.number_input(f"Limit Score {scores[i+1]} (P1)", value=350+(i*50), min_value=350, step=1, format="%d")
            manual_t['P1'].append(float(val))
        
        # P2/P3 Magnetismo e Coercitività
        for label, key, def_val in [("Magnetism (T)", "P2", 0.4), ("Coercivity (T)", "P3", 0.4)]:
            st.markdown(f"**{label}**")
            sf = st.selectbox(f"Tiers {key}", [2, 3, 4, 5], index=1 if key=="P2" else 3, key=f"sf_{key}")
            sc_list = [round((i+1)/sf, 2) for i in range(sf)]
            for i in range(sf - 1):
                v = st.number_input(f"Limit Score {sc_list[i+1]} ({key})", value=def_val+(i*0.2), min_value=def_val, key=f"t_{key}_{i}")
                manual_t[key].append(v)
            if key == "P2": sf_m = sf
            else: sf_c = sf

        st.markdown('<div class="blue-header">2. Weights</div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight T", 0.0, 1.0, 0.33)
        w_p2 = st.slider("Weight M", 0.0, 1.0-w_p1, 0.33)
        w_p3 = round(max(0.0, 1.0 - (w_p1 + w_p2)), 2)

    # --- CALCOLO OPS/OSS ---
    def get_score(df_col, n, thresholds):
        sc_list = [round((i+1)/n, 2) for i in range(n)]
        res = pd.Series(sc_list[0], index=df_col.index)
        for i, t in enumerate(thresholds):
            res[df_col >= t] = sc_list[i+1]
        return res

    p1_s = get_score(df['P1'], sf_t, manual_t['P1'])
    p2_s = get_score(df['P2'], sf_m, manual_t['P2'])
    p3_s = get_score(df['P3'], sf_c, manual_t['P3'])
    
    df['OPS'] = (p1_s**w_p1) * (p2_s**w_p2) * (p3_s**w_p3)
    
    # Sostenibilità (fixed geometric mean S1-S10)
    s_cols = [f'S{i}' for i in range(1, 11)]
    if all(c in df.columns for c in s_cols):
        s_vals = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).clip(lower=1e-3).values
        df['OSS'] = np.exp(np.mean(np.log(s_vals), axis=1))
    else: df['OSS'] = 0.5

    # --- VISUALIZZAZIONE ---
    tab1, tab2 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map"])

    with tab1:
        pts = df[['OPS', 'OSS']].to_numpy()
        efficient = np.ones(pts.shape[0], dtype=bool)
        for i, c in enumerate(pts):
            if efficient[i]: efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))
        df['Status'] = np.where(efficient, 'Optimal', 'Standard')
        
        fig = px.scatter(df, x='OPS', y='OSS', color='Status', hover_name='Material_Name', color_discrete_map={'Optimal': '#1e3a8a', 'Standard': '#cbd5e1'})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']])

    with tab2:
        st.markdown("### Scalability Analysis (Calculated via Atomic Fraction)")
        fig_sc = px.scatter(df, x='Calc_Res', y='Calc_Prod', color='OSS', 
                            size=np.where(df['Status']=='Optimal', 12, 6),
                            hover_name='Material_Name', log_x=True, log_y=True,
                            color_continuous_scale="Viridis",
                            labels={'Calc_Res': 'Reserves (t)', 'Calc_Prod': 'Production (t/yr)'})
        fig_sc.update_layout(template="plotly_white")
        st.plotly_chart(fig_sc, use_container_width=True)

else:
    st.warning("Carica i file 'AF_vectors.csv' e 'Materials Database 1.csv' per attivare la piattaforma.")
