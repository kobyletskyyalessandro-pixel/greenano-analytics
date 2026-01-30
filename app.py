import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAZIONE & STILE (NUCLEAR WHITE/BLUE) ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root { 
        --primary: #1e3a8a;    /* Midnight Blue */
        --secondary: #2563eb;  /* Royal Blue */
        --bg: #f8fafc;         /* Light Background */
        --text: #0f172a;       /* Dark Text */
    }
    
    /* FORZA TEMA CHIARO GLOBALE */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #1e3a8a !important;
    }
    
    html, body, .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: var(--bg); 
        color: var(--text); 
    }
    
    /* --- SIDEBAR: SFONDO BIANCO --- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }
    
    /* TESTI SIDEBAR -> BLU */
    section[data-testid="stSidebar"] label {
        color: #1e3a8a !important; 
        font-weight: 700 !important;
        font-size: 14px;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #1e3a8a !important;
    }
    section[data-testid="stSidebar"] small, 
    section[data-testid="stSidebar"] .caption {
        color: #64748b !important;
    }
    
    /* --- INPUT BOXES (NUCLEAR FIX BIANCO) --- */
    div[data-baseweb="input"] {
        background-color: #ffffff !important; 
        border: 2px solid #1e3a8a !important; 
        border-radius: 8px !important;
        padding: 0px !important;
    }
    
    div[data-baseweb="input"] input {
        background-color: #ffffff !important; 
        color: #1e3a8a !important; /* NUMERI BLU */
        -webkit-text-fill-color: #1e3a8a !important;
        caret-color: #1e3a8a !important;
        font-weight: 800 !important;
        padding-left: 10px !important;
    }

    /* PULSANTI +/- */
    div[data-baseweb="input"] button {
        background-color: #1e3a8a !important; 
        border: none !important;
        height: 100% !important;
        margin: 0 !important;
        width: 30px !important;
    }
    div[data-baseweb="input"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    div[data-baseweb="input"] button:hover {
        background-color: #2563eb !important;
    }

    /* TITOLI */
    h1, h2, h3, h4 { color: #1e3a8a !important; font-weight: 800; }
    
    /* CARD */
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white !important; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0; 
        box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.1);
        color: #1e3a8a !important;
    }
    
    /* BOTTONI */
    div.stButton > button:first-child { 
        background-color: #1e3a8a !important; 
        color: white !important; 
        border-radius: 8px; 
        border: none; 
        padding: 12px 24px; 
        font-weight: 700; 
    }
    div.stButton > button:hover { 
        background-color: #2563eb !important; 
        transform: translateY(-2px); 
    }
    
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER HEADER BLU ---
def blue_pill_header(text, icon=""):
    st.markdown(f"""
    <div style="
        background-color: #1e3a8a; 
        color: #ffffff !important; 
        padding: 10px 18px; 
        border-radius: 8px; 
        margin-bottom: 15px; 
        font-weight: 700; 
        font-size: 15px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: flex; align-items: center; gap: 8px;">
        <span style="opacity:1; color: #ffffff !important;">{icon}</span> 
        <span style="color: #ffffff !important;">{text}</span>
    </div>
    """, unsafe_allow_html=True)

# --- MOTORE DI CALCOLO (LOGICA DEL COLLAB) ---
def calculate_ops(df, thresholds, weights):
    """
    Replica la logica: Score = (Valore / Soglia) ^ Peso
    Se Valore > Soglia -> Score = 1.0 (Saturazione)
    """
    # 1. Normalizza i valori rispetto alle soglie utente
    # Aggiungiamo 1e-9 per evitare divisioni per zero
    s1 = np.where(df['P1'] >= thresholds['P1'], 1.0, df['P1'] / (thresholds['P1'] + 1e-9))
    s2 = np.where(df['P2'] >= thresholds['P2'], 1.0, df['P2'] / (thresholds['P2'] + 1e-9))
    s3 = np.where(df['P3'] >= thresholds['P3'], 1.0, df['P3'] / (thresholds['P3'] + 1e-9))
    
    # 2. Clip per sicurezza (0.001 a 1.0)
    scores = np.column_stack((s1, s2, s3))
    scores = np.clip(scores, 1e-3, 1.0)
    
    # 3. Applica i pesi (Media Geometrica Pesata: S1^w1 * S2^w2 * S3^w3)
    # Questa formula equivale a exp( sum( w * log(s) ) )
    w = np.array(weights)
    if w.sum() > 0: w = w / w.sum() # Normalizza pesi a 1
    
    final_score = np.exp(np.sum(w * np.log(scores), axis=1))
    return final_score

def weighted_geometric_mean(S, w, eps=1e-12):
    S = np.clip(np.asarray(S, dtype=float), eps, 1.0)
    w = np.asarray(w, dtype=float)
    if w.sum() > 0: w = w / w.sum()
    return np.exp(np.sum(w * np.log(S), axis=1))

def pareto_front(points):
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if efficient[i]:
            if np.any(np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)):
                efficient[i] = False
            else:
                efficient[np.all(P[i] >= P, axis=1) & np.any(P[i] > P, axis=1)] = False
    return efficient

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("MF_sustainability_rank.csv")
        # Assicuriamoci che i dati siano numerici
        for col in ['P1', 'P2', 'P3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

# --- APP LAYOUT ---

st.title("Materials Intelligence Platform")

st.markdown("""
<div style="
    background-color: white; 
    padding: 16px; 
    border-left: 6px solid #1e3a8a; 
    border-radius: 6px; 
    box-shadow: 0 2px 8px rgba(30, 58, 138, 0.1); 
    margin-bottom: 25px;">
    <h4 style="color: #1e3a8a !important; margin: 0 0 5px 0;">🚀 Advanced Analytics Module</h4>
    <p style="margin: 0; font-size: 15px; color: #1e3a8a;">
        Re-apply your <b>Performance Thresholds</b> and <b>Coefficients</b> here to generate a live Pareto analysis.
    </p>
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.header("Calculation Engine")
    
    # --- 1. SOGLIE (THRESHOLDS) ---
    with st.sidebar:
        blue_pill_header("1. Performance Thresholds", "🎯")
        st.caption("Values required to reach max score (1.0)")
        
        # Defaults intelligenti dai dati
        max_p1 = float(df['P1'].max()) if 'P1' in df.columns else 1000.0
        max_p2 = float(df['P2'].max()) if 'P2' in df.columns else 5.0
        max_p3 = float(df['P3'].max()) if 'P3' in df.columns else 5.0

        # Input per l'utente (Riapplica le scelte del sito)
        t_p1 = st.number_input(f"P1: Temperature (K)", value=400.0, step=10.0)
        t_p2 = st.number_input(f"P2: Magnetization (T)", value=1.0, step=0.1)
        t_p3 = st.number_input(f"P3: Coercivity (T)", value=1.0, step=0.1)

        st.markdown("<br>", unsafe_allow_html=True)

    # --- 2. COEFFICIENTI (WEIGHTS) ---
    with st.sidebar:
        blue_pill_header("2. Performance Coefficients", "⚖️")
        st.caption("Set the importance (weights) for the calculation.")
        
        w_p1 = st.slider("Coeff. P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Coeff. P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        # Riepilogo Visivo (Sfondo Blu, testo Bianco)
        st.markdown(f"""
        <div style="
            background-color: #1e3a8a; 
            color: #ffffff !important; 
            padding: 10px; 
            border-radius: 8px; 
            margin-top: 10px; 
            text-align: center;
            font-size: 14px;
            font-weight: 500;">
            <span style="color: white !important;">Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.info("🌍 Sustainability (OSS) is fixed by LCA data.")

    # --- MAIN CALCULATION ---
    if all(c in df.columns for c in ['P1', 'P2', 'P3']):
        
        # Ricalcolo OPS Live basato sugli input utente
        thresholds = {'P1': t_p1, 'P2': t_p2, 'P3': t_p3}
        weights_perf = [w_p1, w_p2, w_p3]
        
        df['OPS'] = calculate_ops(df, thresholds, weights_perf)
        
        # OSS (Sustainability) è fisso o preso dal CSV
        s_cols = [f"S{i}" for i in range(1, 11)]
        if all(c in df.columns for c in s_cols):
            S_mat = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
            df['OSS'] = weighted_geometric_mean(S_mat, np.ones(10)/10)
        else:
            df['OSS'] = 0.5

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "🔬 Stability Analysis"])

        # TAB 1: PARETO
        with tab1:
            blue_pill_header("Live Pareto Frontier", "🏆")
            
            colA, colB = st.columns([2, 1])
            with colA:
                # Calcolo Frontiera Pareto su OPS (nuovo) vs OSS
                mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
                df['Status'] = np.where(mask, 'Optimal Choice', 'Sub-optimal')
                
                fig = px.scatter(
                    df, x='OPS', y='OSS', color='Status',
                    hover_name='Material_Name', hover_data=['Chemical_Formula', 'P1', 'P2', 'P3'],
                    color_discrete_map={'Optimal Choice': '#1e3a8a', 'Sub-optimal': '#cbd5e1'},
                    opacity=0.9, size_max=15
                )
                fig.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
                fig.update_layout(
                    template="plotly_white", 
                    xaxis_title="OPS (Performance Score - Calculated)", 
                    yaxis_title="OSS (Sustainability Score)",
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Materials (Based on your inputs)**")
                display_cols = ['Material_Name', 'OPS', 'OSS', 'P1', 'P2', 'P3']
                st.dataframe(
                    df[mask].sort_values(by="OPS", ascending=False)[display_cols], 
                    use_container_width=True, height=500
                )

        # TAB 2: SUPPLY CHAIN
        with tab2:
            blue_pill_header("Supply Chain Criticality", "🏭")
            if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
                fig_scale = px.scatter(
                    df, x='Plong_t', y='Pmax_t_per_yr', color='OSS',
                    log_x=True, log_y=True, hover_name='Material_Name',
                    color_continuous_scale="Viridis",
                    labels={'Plong_t': 'Reserves (tons)', 'Pmax_t_per_yr': 'Production (t/yr)'}
                )
                fig_scale.update_layout(template="plotly_white")
                st.plotly_chart(fig_scale, use_container_width=True)
            else:
                st.warning("Missing Supply Data (Pmax/Plong).")

        # TAB 3: MONTE CARLO
        with tab3:
            blue_pill_header("Robustness & Sensitivity", "🔬")
            st.markdown("Test if your selected material remains optimal if weights vary slightly.")
            
            # Filtra solo i materiali Pareto-efficienti per la selezione
            optimal_materials = df[mask]['Material_Name'].unique()
            if len(optimal_materials) > 0:
                sel_mat = st.selectbox("Select Material:", optimal_materials)
                
                if st.button("Run Simulation ⚡"):
                    idx = df[df['Material_Name'] == sel_mat].index[0]
                    N = 1000
                    rng = np.random.default_rng()
                    
                    # Simula pesi vicini a quelli scelti dall'utente (dirichlet concentrata)
                    alpha = np.array(weights_perf) * 50 + 1
                    W_ops_sim = rng.dirichlet(alpha, N)
                    W_oss_sim = rng.dirichlet(np.ones(10) * 20, N)

                    # Ricostruisci score singoli basati sulle soglie attuali
                    s_single = []
                    for col, thresh in zip(['P1','P2','P3'], [t_p1,t_p2,t_p3]):
                        val = df.loc[idx, col]
                        score = 1.0 if val >= thresh else val / (thresh + 1e-9)
                        s_single.append(max(1e-6, score))
                    s_vec = np.array(s_single)

                    # Calcolo Nuvola
                    cloud_ops = np.exp(np.dot(W_ops_sim, np.log(s_vec)))
                    s_oss_vec = df.loc[idx, s_cols].to_numpy(dtype=float)
                    s_oss_vec = np.clip(s_oss_vec, 1e-6, 1.0)
                    cloud_oss = np.exp(np.dot(W_oss_sim, np.log(s_oss_vec)))

                    fig_mc = px.scatter(x=cloud_ops, y=cloud_oss, opacity=0.3, color_discrete_sequence=['#2563eb'])
                    fig_mc.add_trace(go.Scatter(x=[df.loc[idx,'OPS']], y=[df.loc[idx,'OSS']], mode='markers', 
                                              marker=dict(color='#dc2626', size=15, symbol='star'), name='Your Selection'))
                    fig_mc.update_layout(template="plotly_white", xaxis_title="OPS Stability", yaxis_title="OSS Stability")
                    st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.warning("No optimal materials found with current thresholds.")

    else:
        st.error("Missing columns P1, P2, P3 in CSV. Ensure your data file is correct.")

else:
    st.warning("⚠️ Upload 'MF_sustainability_rank.csv' to GitHub.")
