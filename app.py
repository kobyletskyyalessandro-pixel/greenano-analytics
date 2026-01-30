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
        --bg: #f8fafc;         /* Light Background */
    }
    
    /* RESET GENERALE */
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e3a8a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e3a8a; }
    
    /* SIDEBAR BIANCA */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* INPUT BOX & SELECTBOX: BIANCO, BORDO GRIGIO, TESTO BLU */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important; /* Grigio */
        border-radius: 8px !important;
        color: #1e3a8a !important;
    }
    input[type="number"], div[data-baseweb="select"] span {
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
        caret-color: #1e3a8a !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] svg, div[data-baseweb="input"] button {
        color: #64748b !important;
        fill: #64748b !important;
    }

    /* TITOLI HEADER */
    h1, h2, h3, h4 { color: #1e3a8a !important; font-weight: 800; }
    
    /* CARD PRINCIPALI */
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white !important; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1e3a8a !important;
    }
    
    /* BOTTONI */
    div.stButton > button {
        background-color: #1e3a8a !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    
    /* SIDEBAR SETTINGS TITLE (Semplice testo) */
    .sidebar-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER HEADER (TITOLI BLU NEI RIQUADRI) ---
def blue_header(text):
    st.markdown(f"""
    <div style="
        background-color: #1e3a8a; 
        color: white; 
        padding: 8px 12px; 
        border-radius: 6px; 
        margin-bottom: 10px; 
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        {text}
    </div>
    """, unsafe_allow_html=True)

# --- MOTORE DI CALCOLO (TIER SYSTEM & PRODUCT FORMULA) ---

# Mappa dei punteggi (Score Map) come da Colab
SF_SCORE_MAP = {
    2: [1.0, 0.5],
    3: [1.0, 0.6, 0.3],
    4: [1.0, 0.75, 0.5, 0.25],
    5: [1.0, 0.8, 0.6, 0.4, 0.2]
}

def assign_tiered_scores(df, col_name, sf_value):
    """
    Assegna i punteggi (P_i) basandosi sulla divisione in quartili (Tiers).
    """
    scores_list = SF_SCORE_MAP.get(sf_value, SF_SCORE_MAP[3])
    # Ordina dal migliore al peggiore
    sorted_df = df.sort_values(by=col_name, ascending=False).copy()
    num_rows = len(sorted_df)
    sorted_df['temp_score'] = scores_list[-1] # Default al più basso
    
    # Assegna i punteggi per fasce
    for i in range(sf_value):
        cut_off_start = int(num_rows * i / sf_value)
        cut_off_end = int(num_rows * (i + 1) / sf_value)
        
        # Gestione ultimo blocco per includere eventuali resti
        if i < sf_value - 1:
            sorted_df.iloc[cut_off_start:cut_off_end, sorted_df.columns.get_loc('temp_score')] = scores_list[i]
        else:
            sorted_df.iloc[cut_off_start:, sorted_df.columns.get_loc('temp_score')] = scores_list[i]
            
    return sorted_df['temp_score'].sort_index()

def calculate_ops_tiered(df, tiers_config, weights):
    """
    Calcola OPS = P1^w1 * P2^w2 * P3^w3
    """
    # 1. Calcola i singoli punteggi di performance (P1, P2, P3) basati sui Tiers
    ps1 = assign_tiered_scores(df, 'P1', tiers_config['P1'])
    ps2 = assign_tiered_scores(df, 'P2', tiers_config['P2'])
    ps3 = assign_tiered_scores(df, 'P3', tiers_config['P3'])
    
    # 2. Crea la matrice dei punteggi
    scores = np.column_stack((ps1, ps2, ps3))
    # Clip per evitare log(0)
    scores = np.clip(scores, 1e-3, 1.0)
    
    # 3. Normalizza i pesi (devono sommare a 1)
    w = np.array(weights)
    if w.sum() > 0: w = w / w.sum()
    
    # 4. Calcola il prodotto pesato: exp( sum( w * log(scores) ) )
    # Questo equivale matematicamente a: Score1^w1 * Score2^w2 * Score3^w3
    return np.exp(np.sum(w * np.log(scores), axis=1))

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
        for col in ['P1', 'P2', 'P3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

# --- APP LAYOUT ---

st.title("Materials Intelligence Platform")

st.markdown("""
<div style="padding: 15px; border-left: 5px solid #1e3a8a; background-color: white; margin-bottom: 25px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <h4 style="margin:0; color:#1e3a8a;">🚀 Calculation Engine</h4>
    <p style="margin:0; color:#475569;">Configure Tiers and Weights to rank materials using weighted performance scores.</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
    
    # 1. TIERS
    with st.sidebar:
        blue_header("1. Performance Tiers")
        sf_t = st.selectbox("Tiers for P1 (Temp)", [2, 3, 4, 5], index=2) 
        sf_m = st.selectbox("Tiers for P2 (Mag)", [2, 3, 4, 5], index=1)
        sf_c = st.selectbox("Tiers for P3 (Coerc)", [2, 3, 4, 5], index=3)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. WEIGHTS
    with st.sidebar:
        blue_header("2. Coefficients")
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        # RIASSUNTO: BIANCO, BORDO GRIGIO, SCRITTA BLU
        st.markdown(f"""
        <div style="
            background-color: white; 
            color: #1e3a8a; 
            padding: 10px; 
            border: 1px solid #cbd5e1; 
            border-radius: 8px; 
            text-align: center; 
            margin-top: 10px; 
            font-weight: 600;
            font-size: 14px;">
            Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.caption("Sustainability data is fixed (LCA).")

    # --- CALCOLO ---
    if all(c in df.columns for c in ['P1', 'P2', 'P3']):
        
        tiers_config = {'P1': sf_t, 'P2': sf_m, 'P3': sf_c}
        weights_perf = [w_p1, w_p2, w_p3]
        
        # CALCOLO OPS CON FORMULA: P1^w1 * P2^w2 * P3^w3
        df['OPS'] = calculate_ops_tiered(df, tiers_config, weights_perf)
        
        s_cols = [f"S{i}" for i in range(1, 11)]
        if all(c in df.columns for c in s_cols):
            S_mat = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
            df['OSS'] = weighted_geometric_mean(S_mat, np.ones(10)/10)
        else:
            df['OSS'] = 0.5

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🏆 Ranking", "🏭 Criticality", "🔬 Stability"])

        with tab1:
            colA, colB = st.columns([2, 1])
            with colA:
                mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
                df['Status'] = np.where(mask, 'Optimal', 'Standard')
                
                fig = px.scatter(
                    df, x='OPS', y='OSS', color='Status',
                    hover_name='Material_Name', hover_data=['Chemical_Formula'],
                    color_discrete_map={'Optimal': '#1e3a8a', 'Standard': '#cbd5e1'},
                    opacity=0.9
                )
                fig.update_layout(template="plotly_white", xaxis_title="OPS (Performance Score)", yaxis_title="OSS (Sustainability Score)")
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Materials**")
                # Mostriamo i dati calcolati
                st.dataframe(df[mask].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS', 'P1', 'P2', 'P3']], use_container_width=True, height=500)

        with tab2:
            if 'Pmax_t_per_yr' in df.columns:
                fig_scale = px.scatter(df, x='Plong_t', y='Pmax_t_per_yr', color='OSS', log_x=True, log_y=True, hover_name='Material_Name', color_continuous_scale="Viridis")
                fig_scale.update_layout(template="plotly_white")
                st.plotly_chart(fig_scale, use_container_width=True)
            else:
                st.warning("No Supply Data")

        with tab3:
            st.markdown("Select optimal material to test stability:")
            opts = df[mask]['Material_Name'].unique()
            if len(opts) > 0:
                sel = st.selectbox("Material", opts)
                if st.button("Simulate"):
                    idx = df[df['Material_Name'] == sel].index[0]
                    rng = np.random.default_rng()
                    W_ops = rng.dirichlet(np.array(weights_perf)*50+1, 1000)
                    
                    ps1 = assign_tiered_scores(df, 'P1', sf_t).loc[idx]
                    ps2 = assign_tiered_scores(df, 'P2', sf_m).loc[idx]
                    ps3 = assign_tiered_scores(df, 'P3', sf_c).loc[idx]
                    
                    # Simulazione OPS con variazione pesi
                    c_ops = np.exp(np.dot(W_ops, np.log([ps1, ps2, ps3] + np.array([1e-9]*3))))
                    
                    W_oss = rng.dirichlet(np.ones(10)*20, 1000)
                    s_oss = df.loc[idx, s_cols].to_numpy(dtype=float)
                    c_oss = np.exp(np.dot(W_oss, np.log(s_oss+1e-9)))
                    
                    fig_mc = px.scatter(x=c_ops, y=c_oss, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                    st.plotly_chart(fig_mc, use_container_width=True)

    else:
        st.error("CSV Missing P1/P2/P3 columns")
else:
    st.warning("Upload CSV")
