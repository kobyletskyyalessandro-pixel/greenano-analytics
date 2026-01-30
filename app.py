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

# --- 2. PULIZIA E CARICAMENTO ---
def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^-0.9.]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def load_and_sync_data():
    try:
        df = pd.read_csv("AF_vectors.csv")
        db = pd.read_csv("Materials Database 1.csv")
        db = db.dropna(subset=['Z'])
        db['Z'] = db['Z'].astype(int)
        
        def get_prop_vector(col_keyword):
            col_name = [c for c in db.columns if col_keyword.lower() in c.lower()][0]
            return db.set_index('Z')[col_name].apply(lambda x: clean_numeric(pd.Series([x])).iloc[0]).reindex(range(1, 119)).fillna(0).values

        # Calcolo proprietà vettoriali
        v_prod = get_prop_vector('production')
        v_res = get_prop_vector('reserve')
        v_risk = get_prop_vector('risk')
        
        af_cols = [f'AF_{i}' for i in range(1, 119)]
        af_matrix = df[af_cols].fillna(0).values

        df['Calc_Production'] = af_matrix @ v_prod
        df['Calc_Reserves'] = af_matrix @ v_res
        df['Calc_Supply_Risk'] = af_matrix @ v_risk
        
        # OSS di fallback se mancano S1-S10
        if not all(f'S{i}' in df.columns for i in range(1, 11)):
            def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
            df['OSS'] = 1 - norm(df['Calc_Supply_Risk'])
            
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Errore: {e}")
        return None

# --- 3. LOGICA RANKING ---
def generate_linear_scores(n_tiers):
    return [round((i + 1) / n_tiers, 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, n_tiers, thresholds):
    scores = generate_linear_scores(n_tiers)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    for i in range(len(thresholds)):
        assigned[df[col_name] >= thresholds[i]] = scores[i+1]
    return assigned

# --- 4. APP ---
df = load_and_sync_data()

if df is not None:
    st.sidebar.markdown('<p class="settings-title">Control Panel</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        # Input soglie... (omessi per brevità, stessa logica precedente)
        sf_t = st.selectbox("Subcategories (P1)", [2,3,4,5], index=2)
        # [Inserire qui i loop number_input per P1, P2, P3 come nei codici precedenti]

        st.markdown('<div class="blue-section-header"><p>2. "Cloud" Settings</p></div>', unsafe_allow_html=True)
        # QUESTO È IL SEGRETO PER VEDERE LA NUVOLA
        zoom_level = st.slider("Exclude Top % Abundant (Zoom)", 0.0, 10.0, 2.0, help="Esclude i materiali troppo comuni per vedere quelli rari.")
        point_size = st.slider("Point Size", 2, 12, 4)
        point_opacity = st.slider("Opacity", 0.1, 1.0, 0.5)

    # Calcoli OPS
    # [Logica assign_tiered_scores...]
    df['OPS'] = 0.5 # Placeholder calcolo effettivo

    t1, t2 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map"])

    with t2:
        st.markdown("### Resource Distribution (Linear Scale)")
        
        # FILTRO DINAMICO PER ESPANDERE LA NUVOLA
        q_prod = np.percentile(df['Calc_Production'], 100 - zoom_level)
        q_res = np.percentile(df['Calc_Reserves'], 100 - zoom_level)
        
        df_cloud = df[(df['Calc_Production'] <= q_prod) & (df['Calc_Reserves'] <= q_res)].copy()
        
        fig = px.scatter(df_cloud, 
                         x='Calc_Reserves', 
                         y='Calc_Production',
                         color='OSS',
                         hover_name='Material_Name',
                         hover_data={'Chemical_Formula': True, 'Calc_Reserves': ':.2e', 'Calc_Production': ':.2e'},
                         color_continuous_scale="Viridis",
                         labels={'Calc_Reserves': 'Reserves (t)', 'Calc_Production': 'Production (t/yr)'})
        
        fig.update_traces(marker=dict(size=point_size, opacity=point_opacity, line=dict(width=0)))
        fig.update_layout(template="plotly_white", height=700)
        
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Stai visualizzando il {100-zoom_level}% dei materiali meno abbondanti per permettere alla nuvola di espandersi.")
