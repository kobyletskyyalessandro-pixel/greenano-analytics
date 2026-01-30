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
    
    html, body, .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: var(--bg); 
        color: var(--text); 
    }
    
    /* Sidebar: Sfondo Bianco/Grigio Chiaro */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Card Bianche per i Grafici (Main Area) */
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white; 
        border-radius: 12px; 
        padding: 20px; 
        border: 1px solid #e2e8f0; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Bottoni */
    div.stButton > button:first-child { 
        background-color: var(--primary); 
        color: white !important; 
        border-radius: 8px; 
        border: none; 
        padding: 12px 24px; 
        font-weight: 600; 
    }
    div.stButton > button:hover { 
        background-color: var(--secondary); 
        transform: translateY(-2px); 
    }
    
    /* Rimuovi padding extra */
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER PER TITOLI STILIZZATI (RIQUADRI BLU) ---
def blue_header(text, icon=""):
    st.markdown(f"""
    <div style="
        background-color: #1e3a8a; 
        color: white; 
        padding: 10px 15px; 
        border-radius: 8px; 
        margin-bottom: 15px; 
        font-weight: 700; 
        font-size: 16px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex; align-items: center; gap: 10px;">
        <span>{icon}</span> {text}
    </div>
    """, unsafe_allow_html=True)

# --- 2. MOTORE DI CALCOLO ---

def calculate_ops(df, thresholds, weights):
    s1 = np.where(df['P1'] >= thresholds['P1'], 1.0, df['P1'] / (thresholds['P1'] + 1e-9))
    s2 = np.where(df['P2'] >= thresholds['P2'], 1.0, df['P2'] / (thresholds['P2'] + 1e-9))
    s3 = np.where(df['P3'] >= thresholds['P3'], 1.0, df['P3'] / (thresholds['P3'] + 1e-9))
    scores = np.column_stack((s1, s2, s3))
    scores = np.clip(scores, 1e-6, 1.0)
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

# --- 3. CARICAMENTO DATI ---
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

# --- 4. INTERFACCIA UTENTE ---

# HEADER PRINCIPALE
st.title("Materials Intelligence Platform")

st.markdown("""
<div style="background-color: white; padding: 15px; border-left: 5px solid #1e3a8a; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px;">
    <strong style="color: #1e3a8a;">Scientific Dashboard:</strong> 
    Use the sidebar controls to customize performance thresholds. 
    Sustainability scores are fixed based on LCA methodology.
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR (SFONDO BIANCO, HEADER BLU) ---
    st.sidebar.header("User Controls")
    
    # SEZIONE 1: SOGLIE
    with st.sidebar:
        blue_header("1. Performance Thresholds", "🎯")
        st.caption("Minimum values to achieve max score (1.0)")
        
        max_p1 = float(df['P1'].max()) if 'P1' in df.columns else 1000.0
        max_p2 = float(df['P2'].max()) if 'P2' in df.columns else 1000.0
        max_p3 = float(df['P3'].max()) if 'P3' in df.columns else 1000.0
        
        t_p1 = st.number_input(f"Target P1 (Magnetization)", value=max_p1*0.5, step=10.0)
        t_p2 = st.number_input(f"Target P2 (Anisotropy)", value=max_p2*0.5, step=0.1)
        t_p3 = st.number_input(f"Target P3 (Curie Temp)", value=max_p3*0.5, step=10.0)
        
        st.divider()

    # SEZIONE 2: PESI
    with st.sidebar:
        blue_header("2. Importance Weights", "⚖️")
        
        w_p1 = st.slider("Weight P1", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Weight P2", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        # INFO BOX (BIANCO SU BLU)
        st.markdown(f"""
        <div style="
            background-color: #1e3a8a; 
            color: white; 
            padding: 12px; 
            border-radius: 8px; 
            margin-top: 10px; 
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            P1: {w_p1:.2f} &nbsp;|&nbsp; P2: {w_p2:.2f} &nbsp;|&nbsp; P3: {w_p3:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.info("🌍 Sustainability (OSS) metrics are fixed.")

    # --- CALCOLI ---
    if all(c in df.columns for c in ['P1', 'P2', 'P3']):
        
        # OPS Custom & OSS Fixed
        thresholds = {'P1': t_p1, 'P2': t_p2, 'P3': t_p3}
        weights_perf = [w_p1, w_p2, w_p3]
        df['OPS'] = calculate_ops(df, thresholds, weights_perf)
        
        s_cols = [f"S{i}" for i in range(1, 11)]
        if all(c in df.columns for c in s_cols):
            S_mat = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
            df['OSS'] = weighted_geometric_mean(S_mat, np.ones(10)/10)
        else:
            df['OSS'] = 0.5
            
        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🏆 Custom Ranking", "🏭 Scalability Map", "🔬 Stability Cloud"])

        # TAB 1: RANKING
        with tab1:
            blue_header("Custom Pareto Frontier", "🏆") # Titolo Sezione Blu
            
            colA, colB = st.columns([2, 1])
            with colA:
                mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
                df['Status'] = np.where(mask, 'Best Choice', 'Standard')
                
                fig = px.scatter(
                    df, x='OPS', y='OSS', color='Status',
                    hover_name='Material_Name', hover_data=['Chemical_Formula', 'P1', 'P2', 'P3'],
                    color_discrete_map={'Best Choice': '#1e3a8a', 'Standard': '#cbd5e1'},
                    opacity=0.9
                )
                fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
                fig.update_layout(
                    template="plotly_white", 
                    xaxis_title="OPS (Performance Score)", 
                    yaxis_title="OSS (Sustainability Score)",
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Materials List**")
                st.dataframe(
                    df[mask].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS', 'P1']], 
                    use_container_width=True, height=500
                )

        # TAB 2: SUPPLY CHAIN
        with tab2:
            blue_header("Criticality Analysis", "🏭")
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
            blue_header("Robustness Check", "🔬")
            st.markdown("Select a material to simulate how weight variations affect its position.")
            
            sel_mat = st.selectbox("Select Material:", df[mask]['Material_Name'].unique())
            
            if st.button("Check Stability ⚡"):
                idx = df[df['Material_Name'] == sel_mat].index[0]
                N = 1000
                rng = np.random.default_rng()
                
                alpha = np.array(weights_perf) * 50 + 1
                W_ops_sim = rng.dirichlet(alpha, N)
                W_oss_sim = rng.dirichlet(np.ones(10) * 20, N)

                s_single = []
                for col, thresh in zip(['P1','P2','P3'], [t_p1,t_p2,t_p3]):
                    val = df.loc[idx, col]
                    score = 1.0 if val >= thresh else val / (thresh + 1e-9)
                    s_single.append(max(1e-6, score))
                s_vec = np.array(s_single)

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
        st.error("Missing columns P1, P2, P3 in CSV.")

else:
    st.warning("⚠️ Upload 'MF_sustainability_rank.csv' to GitHub.")
