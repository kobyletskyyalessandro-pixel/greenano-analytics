import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURAZIONE & STILE ---
st.set_page_config(page_title="GreeNano Analytics", layout="wide")

st.markdown("""
    <style>
    :root { --primary: #1e3a8a; }
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    
    .blue-section-header {
        background-color: #1e3a8a;
        padding: 10px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .blue-section-header p {
        color: #ffffff !important;
        margin: 0 !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    .custom-summary-box {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: #1e3a8a;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTORE DI CALCOLO ---

def generate_linear_scores(n_tiers):
    """Genera score decrescenti: 1.0 per il tier migliore, etc."""
    step = 1.0 / n_tiers
    return [round(1.0 - (i * step), 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, sf_value, manual_thresholds):
    scores_list = generate_linear_scores(sf_value)
    assigned_scores = pd.Series(scores_list[-1], index=df.index, dtype=float)
    
    # Ordiniamo le soglie dal Tier 1 (più alto) a scendere
    # Applicazione: chi supera la soglia del Tier 1 prende score[0], e così via.
    # Per non sovrascrivere i valori alti con quelli bassi, applichiamo in ordine inverso (dal peggiore al migliore)
    
    thresholds_with_scores = list(zip(manual_thresholds, scores_list[:-1]))
    # Ordiniamo per valore di soglia crescente
    thresholds_with_scores.sort(key=lambda x: x[0])
    
    for thresh, score in thresholds_with_scores:
        assigned_scores[df[col_name] >= thresh] = score
        
    return assigned_scores

# --- APP ---
st.title("Materials Intelligence Platform")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("MF_sustainability_rank.csv")
        for col in ['P1', 'P2', 'P3']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except: return None

df = load_data()

if df is not None:
    st.sidebar.markdown('<p style="font-size:20px; font-weight:700; color:#1e3a8a;">Settings</p>', unsafe_allow_html=True)
    
    manual_thresholds = {'P1': [], 'P2': [], 'P3': []}
    
    with st.sidebar:
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        
        # P1: TEMP (Min 350)
        st.markdown("**P1: Temperature (K)**")
        sf_t = st.selectbox("Tiers (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Tier {i+1} Limit (Score {sc_t[i]})", value=350.0 + (100.0/(i+1)), min_value=350.0, key=f"p1_{i}")
            manual_thresholds['P1'].append(val)

        # P2: MAG (Min 0.4)
        st.markdown("---")
        st.markdown("**P2: Magnetization (T)**")
        sf_m = st.selectbox("Tiers (P2)", [2, 3, 4, 5], index=1)
        sc_m = generate_linear_scores(sf_m)
        for i in range(sf_m - 1):
            val = st.number_input(f"Tier {i+1} Limit (Score {sc_m[i]})", value=0.4 + (0.2/(i+1)), min_value=0.4, key=f"p2_{i}")
            manual_thresholds['P2'].append(val)

        # P3: COERC (Min 0.4)
        st.markdown("---")
        st.markdown("**P3: Coercivity (T)**")
        sf_c = st.selectbox("Tiers (P3)", [2, 3, 4, 5], index=3)
        sc_c = generate_linear_scores(sf_c)
        for i in range(sf_c - 1):
            val = st.number_input(f"Tier {i+1} Limit (Score {sc_c[i]})", value=0.4 + (0.2/(i+1)), min_value=0.4, key=f"p3_{i}")
            manual_thresholds['P3'].append(val)

        st.markdown('<div class="blue-section-header"><p>2. Performance Coefficients</p></div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight P1", 0.0, 1.0, 0.33)
        w_p2 = st.slider("Weight P2", 0.0, 1.0 - w_p1, 0.33)
        w_p3 = round(1.0 - (w_p1 + w_p2), 2)
        
        st.markdown(f'<div class="custom-summary-box">T: {w_p1:.2f} | M: {w_p2:.2f} | C: {w_p3:.2f}</div>', unsafe_allow_html=True)

    # --- CALCOLO OPS ---
    ps1 = assign_tiered_scores(df, 'P1', sf_t, manual_thresholds['P1'])
    ps2 = assign_tiered_scores(df, 'P2', sf_m, manual_thresholds['P2'])
    ps3 = assign_tiered_scores(df, 'P3', sf_c, manual_thresholds['P3'])
    
    df['OPS'] = np.power(ps1, w_p1) * np.power(ps2, w_p2) * np.power(ps3, w_p3)
    
    # OSS Semplificato (Media Geometrica S1-S10)
    s_cols = [f'S{i}' for i in range(1, 11)]
    df['OSS'] = np.exp(np.mean(np.log(df[s_cols].clip(lower=1e-3)), axis=1))

    # --- VISUALIZZAZIONE ---
    t1, t2 = st.tabs(["🏆 Pareto Ranking", "📊 Data Table"])
    with t1:
        fig = px.scatter(df, x='OPS', y='OSS', hover_name='Material_Name', color='OPS', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.dataframe(df[['Material_Name', 'OPS', 'OSS', 'P1', 'P2', 'P3']].sort_values(by='OPS', ascending=False))
