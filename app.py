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
        --secondary: #2563eb;  /* Royal Blue */
        --bg: #f8fafc;         /* Light Background */
        --text: #0f172a;       /* Dark Text */
    }
    
    /* RESET GLOBALE: APP BIANCA, TESTO BLU */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        color: #1e3a8a;
    }
    html, body, .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: #f8fafc; 
        color: #1e3a8a; 
    }
    
    /* --- SIDEBAR: SFONDO BIANCO --- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }
    
    /* TESTI SIDEBAR STANDARD -> BLU */
    section[data-testid="stSidebar"] label {
        color: #1e3a8a !important; 
        font-weight: 700 !important;
        font-size: 14px;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {
        color: #1e3a8a !important;
    }
    section[data-testid="stSidebar"] small, 
    section[data-testid="stSidebar"] .caption {
        color: #64748b !important;
    }
    
    /* --- INPUT BOXES (BIANCO PURO, BORDO BLU, NUMERI BLU) --- */
    div[data-baseweb="input"] {
        background-color: #ffffff !important; 
        border: 2px solid #1e3a8a !important; 
        border-radius: 8px !important;
        padding: 0px !important;
    }
    div[data-baseweb="input"] input {
        background-color: #ffffff !important; 
        color: #1e3a8a !important; 
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

    /* --- NUCLEAR FIX PER SCRITTE BIANCHE NEI BOX BLU --- */
    /* Qualsiasi cosa dentro un elemento con classe 'force-white-container' diventa bianca */
    .force-white-container {
        color: #ffffff !important;
    }
    .force-white-container * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* TITOLI & CARD */
    h1, h2, h3, h4 { color: #1e3a8a !important; font-weight: 800; }
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

# --- HELPER HEADER BLU (CON CLASSE NUCLEAR FIX) ---
def blue_pill_header(text, icon=""):
    st.markdown(f"""
    <div class="force-white-container" style="
        background-color: #1e3a8a; 
        padding: 10px 18px; 
        border-radius: 8px; 
        margin-bottom: 15px; 
        font-weight: 700; 
        font-size: 15px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: flex; align-items: center; gap: 8px;">
        <span>{icon}</span> 
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)

# --- MOTORE DI CALCOLO (TIER SYSTEM) ---

SF_SCORE_MAP = {
    2: [1.0, 0.5],
    3: [1.0, 0.6, 0.3],
    4: [1.0, 0.75, 0.5, 0.25],
    5: [1.0, 0.8, 0.6, 0.4, 0.2]
}

def assign_tiered_scores(df, col_name, sf_value):
    scores_list = SF_SCORE_MAP.get(sf_value, SF_SCORE_MAP[3])
    sorted_df = df.sort_values(by=col_name, ascending=False).copy()
    num_rows = len(sorted_df)
    sorted_df['temp_score'] = scores_list[-1]
    
    for i in range(sf_value):
        cut_off_start = int(num_rows * i / sf_value)
        cut_off_end = int(num_rows * (i + 1) / sf_value)
        if i < sf_value - 1:
            sorted_df.iloc[cut_off_start:cut_off_end, sorted_df.columns.get_loc('temp_score')] = scores_list[i]
        else:
            sorted_df.iloc[cut_off_start:, sorted_df.columns.get_loc('temp_score')] = scores_list[i]
    return sorted_df['temp_score'].sort_index()

def calculate_ops_tiered(df, tiers_config, weights):
    ps1 = assign_tiered_scores(df, 'P1', tiers_config['P1'])
    ps2 = assign_tiered_scores(df, 'P2', tiers_config['P2'])
    ps3 = assign_tiered_scores(df, 'P3', tiers_config['P3'])
    
    scores = np.column_stack((ps1, ps2, ps3))
    scores = np.clip(scores, 1e-3, 1.0)
    w = np.array(weights)
    if w.sum() > 0: w = w / w.sum()
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
<div style="
    background-color: white; 
    padding: 16px; 
    border-left: 6px solid #1e3a8a; 
    border-radius: 6px; 
    box-shadow: 0 2px 8px rgba(30, 58, 138, 0.1); 
    margin-bottom: 25px;">
    <h4 style="color: #1e3a8a !important; margin: 0 0 5px 0;">🚀 Advanced Analytics Module</h4>
    <p style="margin: 0; font-size: 15px; color: #1e3a8a;">
        Configure <b>Subcategories (Tiers)</b> and <b>Coefficients</b> to rank materials using the scientific quantile scoring method.
    </p>
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.header("Calculation Engine")
    
    # --- 1. SOGLIE (TIERS) ---
    with st.sidebar:
        blue_pill_header("1. Performance Tiers", "📊")
        st.caption("Select number of subcategories (levels) for ranking.")
        
        sf_t = st.selectbox("Tiers for P1 (Temp)", [2, 3, 4, 5], index=2) 
        sf_m = st.selectbox("Tiers for P2 (Mag)", [2, 3, 4, 5], index=1)
        sf_c = st.selectbox("Tiers for P3 (Coerc)", [2, 3, 4, 5], index=3)

        st.markdown("<br>", unsafe_allow_html=True)

    # --- 2. COEFFICIENTI (WEIGHTS) ---
    with st.sidebar:
        blue_pill_header("2. Performance Coefficients", "⚖️")
        st.caption("Set the importance weights (x, y, z).")
        
        w_p1 = st.slider("Coeff. P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Coeff. P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        # Riepilogo Visivo (CON CLASSE NUCLEAR FIX)
        st.markdown(f"""
        <div class="force-white-container" style="
            background-color: #1e3a8a; 
            padding: 10px; 
            border-radius: 8px; 
            margin-top: 10px; 
            text-align: center;
            font-size: 14px;
            font-weight: 500;">
            Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.info("🌍 Sustainability (OSS) is fixed by LCA data.")

    # --- MAIN CALCULATION ---
    if all(c in df.columns for c in ['P1', 'P2', 'P3']):
        
        tiers_config = {'P1': sf_t, 'P2': sf_m, 'P3': sf_c}
        weights_perf = [w_p1, w_p2, w_p3]
        
        df['OPS'] = calculate_ops_tiered(df, tiers_config, weights_perf)
        
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
                    xaxis_title="OPS (Performance Score - Tiered)", 
                    yaxis_title="OSS (Sustainability Score)",
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Materials**")
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
            st.markdown("Test ranking stability against small weight variations.")
            
            optimal_materials = df[mask]['Material_Name'].unique()
            if len(optimal_materials) > 0:
                sel_mat = st.selectbox("Select Material:", optimal_materials)
                
                if st.button("Run Simulation ⚡"):
                    idx = df[df['Material_Name'] == sel_mat].index[0]
                    N = 1000
                    rng = np.random.default_rng()
                    
                    alpha = np.array(weights_perf) * 50 + 1
                    W_ops_sim = rng.dirichlet(alpha, N)
                    
                    ps1_val = assign_tiered_scores(df, 'P1', sf_t).loc[idx]
                    ps2_val = assign_tiered_scores(df, 'P2', sf_m).loc[idx]
                    ps3_val = assign_tiered_scores(df, 'P3', sf_c).loc[idx]
                    s_vec = np.array([ps1_val, ps2_val, ps3_val])

                    cloud_ops = np.exp(np.dot(W_ops_sim, np.log(s_vec + 1e-9)))
                    
                    W_oss_sim = rng.dirichlet(np.ones(10) * 20, N)
                    s_oss_vec = df.loc[idx, s_cols].to_numpy(dtype=float)
                    s_oss_vec = np.clip(s_oss_vec, 1e-6, 1.0)
                    cloud_oss = np.exp(np.dot(W_oss_sim, np.log(s_oss_vec)))

                    fig_mc = px.scatter(x=cloud_ops, y=cloud_oss, opacity=0.3, color_discrete_sequence=['#2563eb'])
                    fig_mc.add_trace(go.Scatter(x=[df.loc[idx,'OPS']], y=[df.loc[idx,'OSS']], mode='markers', 
                                              marker=dict(color='#dc2626', size=15, symbol='star'), name='Your Selection'))
                    fig_mc.update_layout(template="plotly_white", xaxis_title="OPS Stability", yaxis_title="OSS Stability")
                    st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.warning("No optimal materials found.")

    else:
        st.error("Missing columns P1, P2, P3 in CSV.")

else:
    st.warning("⚠️ Upload 'MF_sustainability_rank.csv' to GitHub.")
