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
    .blue-section-header p { color: #ffffff !important; margin: 0 !important; font-weight: 700 !important; }
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

def clean_val(x):
    """Pulisce stringhe numeriche sporche (es. '1,000.50 tons', 'primary', ecc)"""
    if pd.isna(x) or x == "": return 0.0
    s = str(x).replace(',', '').strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

@st.cache_data
def load_and_process_data():
    try:
        # Carica AF_vectors (Contiene P1, P2, P3 e AF_1..AF_118)
        df = pd.read_csv("AF_vectors.csv")
        
        # Carica Database Elementare (Saltiamo le prime 2 righe se necessario per le intestazioni)
        db = pd.read_csv("Materials Database 1.csv", header=0)
        
        # Individua colonne target (World production, World reserve, Risk, HHI, ESG)
        # Nota: Usiamo indici o nomi flessibili per evitare errori di intestazioni multiple
        db['Z_clean'] = pd.to_numeric(db['Z'], errors='coerce')
        db = db.dropna(subset=['Z_clean'])
        db['Z_clean'] = db['Z_clean'].astype(int)
        
        # Mappatura Proprietà Elementari (Z -> Valore)
        # Cerchiamo di pulire i dati nelle colonne specifiche
        db['P_Val'] = db['World production (tons per year)'].apply(clean_val)
        db['R_Val'] = db['World reserve (tons)'].apply(clean_val)
        db['S_Val'] = db['Supply risk'].apply(clean_val)
        db['H_Val'] = db['HHI'].apply(clean_val)
        db['E_Val'] = db['ESG'].apply(clean_val)

        def get_vec(col_name):
            return db.set_index('Z_clean')[col_name].reindex(range(1, 119)).fillna(0).values

        v_p = get_vec('P_Val')
        v_r = get_vec('R_Val')
        v_s = get_vec('S_Val')
        v_h = get_vec('H_Val')
        v_e = get_vec('E_Val')

        # Matrice frazioni atomiche (Materiali x 118 elementi)
        af_cols = [f'AF_{i}' for i in range(1, 119)]
        af_matrix = df[af_cols].fillna(0).values

        # Calcolo proprietà del materiale (Dot Product)
        df['Calc_Production'] = af_matrix @ v_p
        df['Calc_Reserves'] = af_matrix @ v_r
        df['Calc_Supply_Risk'] = af_matrix @ v_s
        df['Calc_HHI'] = af_matrix @ v_h
        df['Calc_ESG'] = af_matrix @ v_e

        # Assicuriamoci che P1, P2, P3 siano numerici
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Errore caricamento database: {e}")
        return None

# --- 3. LOGICA CALCOLO ---

def generate_tiered_scores(n):
    return [round((i + 1) / n, 2) for i in range(n)]

def assign_manual_scores(df, col, n, thresh_list):
    scores = generate_tiered_scores(n)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    # Ordiniamo thresh per applicazione ascendente (il valore più alto vince)
    for i, val in enumerate(thresh_list):
        assigned[df[col] >= val] = scores[i+1]
    return assigned

# --- 4. UI ---
df = load_and_process_data()

if df is not None:
    st.sidebar.markdown('<p class="settings-title">Settings</p>', unsafe_allow_html=True)
    manual_t = {'P1': [], 'P2': [], 'P3': []}
    is_valid = True
    
    with st.sidebar:
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        
        # P1 TEMP
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_tiered_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Threshold Score {sc_t[i+1]} (P1)", value=int(350+(i*50)), min_value=350, step=1, format="%d", key=f"p1_{i}")
            manual_t['P1'].append(float(val))
        
        # P2 MAG / P3 COERC
        for label, key, d_idx, d_val in [("Magnetization (T)", "P2", 1, 0.4), ("Coercivity (T)", "P3", 3, 0.4)]:
            st.markdown(f"**{label}**")
            sf = st.selectbox(f"Subcategories ({key})", [2, 3, 4, 5], index=d_idx, key=f"sf_{key}")
            sc = generate_tiered_scores(sf)
            for i in range(sf - 1):
                v = st.number_input(f"Threshold Score {sc[i+1]} ({key})", value=d_val+(i*0.2), min_value=d_val, key=f"t_{key}_{i}")
                manual_t[key].append(v)
            if key == "P2": sf_m = sf
            else: sf_c = sf

        # Validazione
        for k in ['P1', 'P2', 'P3']:
            if any(manual_t[k][i] >= manual_t[k][i+1] for i in range(len(manual_t[k])-1)):
                st.error(f"Error: {k} thresholds must be ascending!")
                is_valid = False

        st.markdown('<div class="blue-section-header"><p>2. Coefficients</p></div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight P1", 0.0, 1.0, 0.33)
        w_p2 = st.slider("Weight P2", 0.0, 1.0-w_p1, 0.33)
        w_p3 = round(max(0.0, 1.0 - (w_p1 + w_p2)), 2)

        st.markdown(f'<div class="custom-summary-box"><p style="margin:0; font-size:12px;">P1: {w_p1:.2f} | P2: {w_p2:.2f} | P3: {w_p3:.2f}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="blue-section-header"><p>3. Scalability View</p></div>', unsafe_allow_html=True)
        color_metric = st.selectbox("Coloring Metric", ["OSS", "Supply Risk", "HHI", "ESG"])
        metric_map = {"OSS": "OSS", "Supply Risk": "Calc_Supply_Risk", "HHI": "Calc_HHI", "ESG": "Calc_ESG"}

    if is_valid:
        # Calcoli
        p1_s = assign_manual_scores(df, 'P1', sf_t, manual_t['P1'])
        p2_s = assign_manual_scores(df, 'P2', sf_m, manual_t['P2'])
        p3_s = assign_manual_scores(df, 'P3', sf_c, manual_t['P3'])
        df['OPS'] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)
        
        # OSS (Fisso)
        s_cols = [f'S{i}' for i in range(1, 11)]
        if all(c in df.columns for c in s_cols):
            s_data = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
            df['OSS'] = np.exp(np.mean(np.log(np.clip(s_data, 1e-3, 1.0)), axis=1))
        else: df['OSS'] = 0.5

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
                st.markdown("**Top Materials**")
                st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], use_container_width=True, height=500)

        with t2:
            st.markdown("### Production Capacity vs. Global Reserves")
            st.caption(f"Coloring by: {color_metric}")
            
            # Log scale cleaning
            df_p = df.copy()
            df_p['Calc_Production'] = df_p['Calc_Production'].clip(lower=1e-1)
            df_p['Calc_Reserves'] = df_p['Calc_Reserves'].clip(lower=1e-1)
            
            fig_sc = px.scatter(df_p, x='Calc_Reserves', y='Calc_Production', 
                                color=metric_map[color_metric],
                                size=np.where(efficient, 15, 8),
                                symbol=np.where(efficient, 'star', 'circle'),
                                hover_name='Material_Name',
                                hover_data=['Chemical_Formula', 'Calc_Supply_Risk', 'Calc_HHI'],
                                log_x=True, log_y=True,
                                color_continuous_scale="Viridis",
                                labels={'Calc_Reserves': 'Total Reserves (t)', 'Calc_Production': 'Yearly Production (t/yr)'})
            fig_sc.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig_sc, use_container_width=True)

        with t3:
            opts = df[efficient]['Material_Name'].unique()
            if len(opts) > 0:
                sel = st.selectbox("Select material:", opts)
                if st.button("Run Simulation ⚡"):
                    idx = df[df['Material_Name'] == sel].index[0]
                    rng = np.random.default_rng()
                    W_sim = rng.dirichlet(np.array([w_p1, w_p2, w_p3])*50 + 1, 1000)
                    s_v = np.array([p1_s[idx], p2_s[idx], p3_s[idx]])
                    c_ops = np.exp(np.dot(W_sim, np.log(s_v + 1e-9)))
                    fig_mc = px.scatter(x=c_ops, y=[df.loc[idx, 'OSS']]*1000, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                    fig_mc.add_trace(go.Scatter(x=[df.loc[idx, 'OPS']], y=[df.loc[idx, 'OSS']], mode='markers', marker=dict(color='red', size=12, symbol='star')))
                    st.plotly_chart(fig_mc, use_container_width=True)
else:
    st.error("Assicurati di avere 'AF_vectors.csv' e 'Materials Database 1.csv' nella stessa cartella.")
