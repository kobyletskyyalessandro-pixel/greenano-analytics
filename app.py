import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAZIONE & STILE ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root { 
        --primary: #1e3a8a;    /* Midnight Blue */
        --bg: #f8fafc; 
    }
    
    /* RESET GENERALE */
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* SIDEBAR BIANCA */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* TESTO SETTINGS SEMPLICE */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 15px;
        padding-left: 5px;
    }

    /* BOX TITOLI SEZIONI (BLU CON TESTO BIANCO FORZATO) */
    .blue-section-header {
        background-color: #1e3a8a;
        padding: 10px 15px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .blue-section-header p {
        color: #ffffff !important;
        margin: 0 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    /* ELEMENT CONTAINERS (INPUT, SELECT, SUMMARY): BIANCO CON BORDO GRIGIO */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    .custom-summary-box {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important; /* Grigio default */
        border-radius: 8px !important;
        color: #1e3a8a !important;
    }
    
    /* TESTO BLU DENTRO I BOX BIANCHI */
    input, span, .custom-summary-box p {
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
        font-weight: 600;
    }

    /* BOTTONI +/- DEGLI INPUT */
    div[data-baseweb="input"] button {
        background-color: #f1f5f9 !important;
        color: #1e3a8a !important;
        border: none !important;
    }
    
    /* LABEL STANDARD */
    section[data-testid="stSidebar"] label {
        color: #1e3a8a !important;
        font-weight: 700;
    }

    /* CARD PRINCIPALI */
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white !important; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER UI ---
def section_header(text):
    st.markdown(f'<div class="blue-section-header"><p>{text}</p></div>', unsafe_allow_html=True)

# --- MOTORE DI CALCOLO ---

def generate_linear_scores(n_tiers):
    """Calcola i coefficienti dividendo 1 per il numero di sottocategorie."""
    step = 1.0 / n_tiers
    return [round(1.0 - (i * step), 2) for i in range(n_tiers)]

def assign_manual_tiered_scores(df, col_name, sf_value, manual_thresholds):
    scores_list = generate_linear_scores(sf_value)
    assigned_scores = pd.Series(scores_list[-1], index=df.index, dtype=float)
    
    # Applichiamo le soglie: le ordiniamo per non sovrascrivere male
    sorted_indices = np.argsort(manual_thresholds) 
    for idx in sorted_indices:
        thresh = manual_thresholds[idx]
        score = scores_list[idx]
        assigned_scores[df[col_name] >= thresh] = score
    return assigned_scores

def calculate_ops_product(df, thresholds_map, tiers_config, weights):
    """OPS = P1^w1 * P2^w2 * P3^w3"""
    ps1 = assign_manual_tiered_scores(df, 'P1', tiers_config['P1'], thresholds_map['P1'])
    ps2 = assign_manual_tiered_scores(df, 'P2', tiers_config['P2'], thresholds_map['P2'])
    ps3 = assign_manual_tiered_scores(df, 'P3', tiers_config['P3'], thresholds_map['P3'])
    
    w = np.array(weights)
    if w.sum() > 0: w = w / w.sum()
    
    # Prodotto delle potenze
    return np.power(ps1, w[0]) * np.power(ps2, w[1]) * np.power(ps3, w[2])

# --- CARICAMENTO DATI ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("MF_sustainability_rank.csv")
        for col in ['P1', 'P2', 'P3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except: return None

# --- APP ---
st.title("Materials Intelligence Platform")

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.markdown('<p class="settings-title">Settings</p>', unsafe_allow_html=True)
    
    manual_thresholds = {'P1': [], 'P2': [], 'P3': []}
    
    with st.sidebar:
        section_header("1. Performance Tiers")
        
        # P1: TEMP (Soglia 1 = 350)
        st.markdown("**P1: Temperature (K)**")
        sf_t = st.selectbox("Number of Tiers (P1)", [2, 3, 4, 5], index=2)
        scores_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            # Prima soglia a 350, le altre scalano di 50
            def_v = 350.0 if i == 0 else 350.0 - (i * 50.0)
            val = st.number_input(f"Tier {i+1} Limit (Score {scores_t[i]})", value=def_v, key=f"t1_{i}")
            manual_thresholds['P1'].append(val)
        
        # P2: MAG (Soglia 1 = 0.4)
        st.markdown("---")
        st.markdown("**P2: Magnetization (T)**")
        sf_m = st.selectbox("Number of Tiers (P2)", [2, 3, 4, 5], index=1)
        scores_m = generate_linear_scores(sf_m)
        for i in range(sf_m - 1):
            def_v = 0.4 if i == 0 else 0.4 - (i * 0.1)
            val = st.number_input(f"Tier {i+1} Limit (Score {scores_m[i]})", value=max(0.0, def_v), key=f"t2_{i}")
            manual_thresholds['P2'].append(val)

        # P3: COERC (Soglia 1 = 0.4)
        st.markdown("---")
        st.markdown("**P3: Coercivity (T)**")
        sf_c = st.selectbox("Number of Tiers (P3)", [2, 3, 4, 5], index=3)
        scores_c = generate_linear_scores(sf_c)
        for i in range(sf_c - 1):
            def_v = 0.4 if i == 0 else 0.4 - (i * 0.1)
            val = st.number_input(f"Tier {i+1} Limit (Score {scores_c[i]})", value=max(0.0, def_v), key=f"t3_{i}")
            manual_thresholds['P3'].append(val)

        section_header("2. Performance Coefficients")
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        # SUMMARY BOX (BIANCO, BORDO GRIGIO, TESTO BLU)
        st.markdown(f"""
        <div class="custom-summary-box" style="padding: 10px; text-align: center; margin-top: 10px;">
            <p style="margin:0; font-size: 14px;">
                Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- CALCOLO ---
    tiers_config = {'P1': sf_t, 'P2': sf_m, 'P3': sf_c}
    weights_perf = [w_p1, w_p2, w_p3]
    df['OPS'] = calculate_ops_product(df, manual_thresholds, tiers_config, weights_perf)
    
    # Sustainability (OSS) fixed
    s_cols = [f"S{i}" for i in range(1, 11)]
    if all(c in df.columns for c in s_cols):
        S_mat = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
        df['OSS'] = np.exp(np.mean(np.log(np.clip(S_mat, 1e-3, 1.0)), axis=1))
    else: df['OSS'] = 0.5

    # --- TABS ---
    t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Supply Chain", "🔬 Stability"])

    with t1:
        mask = df[['OPS', 'OSS']].to_numpy()
        # Calcolo Pareto semplificato per visualizzazione
        efficient = np.ones(len(df), dtype=bool)
        for i, c in enumerate(mask):
            if np.any(np.all(mask >= c, axis=1) & np.any(mask > c, axis=1)):
                efficient[i] = False
        df['Status'] = np.where(efficient, 'Optimal', 'Standard')
        
        fig = px.scatter(df, x='OPS', y='OSS', color='Status', hover_name='Material_Name',
                         color_discrete_map={'Optimal': '#1e3a8a', 'Standard': '#cbd5e1'})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], use_container_width=True)

else:
    st.error("Upload 'MF_sustainability_rank.csv' to proceed.")
