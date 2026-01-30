import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAZIONE & STILE (BLUE THEME) ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root { --primary: #1e3a8a; --secondary: #2563eb; --bg: #f8fafc; --text: #0f172a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); }
    h1, h2, h3 { color: var(--primary) !important; font-weight: 700; }
    div.stButton > button:first-child { background-color: var(--primary); color: white; border-radius: 8px; border: none; padding: 12px 24px; font-weight: 600; }
    div.stButton > button:hover { background-color: var(--secondary); transform: translateY(-2px); color: white; }
    div[data-testid="stVerticalBlock"] > div { background-color: white; border-radius: 12px; padding: 15px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTORE DI CALCOLO ---
def weighted_geometric_mean(S, w, eps=1e-12):
    S = np.clip(np.asarray(S, dtype=float), eps, 1.0)
    w = np.asarray(w, dtype=float); w = w/w.sum() if w.sum()>0 else w
    return np.exp(np.sum(w * np.log(S), axis=1))

def pareto_front(points):
    P = np.asarray(points, dtype=float); n = P.shape[0]; efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if efficient[i]:
            if np.any(np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)): efficient[i] = False
            else: efficient[np.all(P[i] >= P, axis=1) & np.any(P[i] > P, axis=1)] = False
    return efficient

@st.cache_data(ttl=3600)
def load_data():
    try: return pd.read_csv("MF_sustainability_rank.csv")
    except: return None

# --- 3. INTERFACCIA ---
st.title("Materials Intelligence Platform")
st.markdown("**Scientific Dashboard**: Explore performance rankings, supply chain risks, and market forecasts.")

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Model Parameters")
    w_p1 = st.sidebar.slider("Weight P1 (Magnetization)", 0.0, 1.0, 0.33)
    rem = 1.0 - w_p1
    w_p2 = st.sidebar.slider("Weight P2 (Anisotropy)", 0.0, max(0.0, rem), min(0.33, rem))
    w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
    
    st.sidebar.info(f"**Current Weights:**\nP1: {w_p1:.2f} | P2: {w_p2:.2f} | P3: {w_p3:.2f}")
    w_oss = st.sidebar.slider("Sustainability Balance", 0.0, 1.0, 0.5)

    # --- CALCOLO REAL-TIME ---
    P_sc = df[["P1","P2","P3"]].fillna(0.3).to_numpy()
    S_sc = df[[f"S{i}" for i in range(1,11)]].fillna(0.3).to_numpy()
    df['OPS'] = weighted_geometric_mean(P_sc, [w_p1,w_p2,w_p3])
    df['OSS'] = weighted_geometric_mean(S_sc, np.ones(10)/10)

    # --- TABS PRINCIPALI ---
    tab1, tab2, tab3 = st.tabs(["🏆 Ranking & Pareto", "🌍 Supply Chain (Paper Figures)", "🔬 Stability Analysis"])

    # TAB 1: RANKING (Classico)
    with tab1:
        colA, colB = st.columns([2,1])
        with colA:
            st.subheader("Performance vs Sustainability")
            mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
            df['Pareto'] = np.where(mask, 'Efficient Frontier', 'Standard')
            fig = px.scatter(df, x='OPS', y='OSS', color='Pareto', hover_name='Material_Name',
                             color_discrete_map={'Efficient Frontier':'#1e3a8a', 'Standard':'#93c5fd'},
                             title="Pareto Optimization")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.subheader("Top Candidates")
            st.dataframe(df[mask].sort_values(by="OPS", ascending=False)[['Material_Name','OPS','OSS']], use_container_width=True)

    # TAB 2: SUPPLY CHAIN (Funzionalità FiguresPaper)
    with tab2:
        st.markdown("### 🏭 Supply Chain & Criticality Analysis")
        st.markdown("Visualizations from *FiguresPaper.ipynb* replicating the scientific methodology.")
        
        if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
            # 1. SCALABILITY MAP (Log-Log)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Fig 1. Scalability Map (Weakest-Link)")
                st.markdown("Identifies materials limited by production (y-axis) or reserves (x-axis).")
                
                fig_scale = px.scatter(
                    df, x='Plong_t', y='Pmax_t_per_yr', color='OSS',
                    log_x=True, log_y=True, hover_name='Material_Name',
                    color_continuous_scale="Viridis", # Come nel paper scientifico
                    labels={'Plong_t': 'Long-term Reserves (tons)', 'Pmax_t_per_yr': 'Max Yearly Prod (t/yr)'}
                )
                st.plotly_chart(fig_scale, use_container_width=True)

            # 2. CORRELATION PLOT (Forecast/Availability)
            with col2:
                st.subheader("Fig 2. Sustainability vs Availability")
                st.markdown("Analysis of correlation between Sustainability (OSS) and Supply Security Score.")
                
                # Calcolo High-Right Score (H) dal Paper
                # H = log(Pmax) + log(Plong)
                # Filtriamo valori validi (>0)
                valid_df = df[(df['Pmax_t_per_yr'] > 0) & (df['Plong_t'] > 0)].copy()
                valid_df['H_Score'] = np.log10(valid_df['Pmax_t_per_yr']) + np.log10(valid_df['Plong_t'])
                
                fig_corr = px.scatter(
                    valid_df, x='H_Score', y='OSS', hover_name='Material_Name',
                    trendline="ols", # Aggiunge la linea di regressione
                    trendline_color_override="#dc2626",
                    labels={'H_Score': 'Supply Security Index (Log Pmax + Log Plong)', 'OSS': 'Sustainability Score'}
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Calcolo Correlazione
                corr = valid_df['H_Score'].corr(valid_df['OSS'])
                st.info(f"📊 **Correlation Coefficient:** {corr:.3f} (Does green mean abundant?)")

        else:
            st.warning("⚠️ Missing 'Pmax' and 'Plong' columns. Ensure your CSV includes supply data.")

    # TAB 3: MONTE CARLO (Stability)
    with tab3:
        st.header("Monte Carlo Uncertainty Simulation")
        sel_mat = st.selectbox("Select Material:", df['Material_Name'].unique())
        if st.button("Run Simulation ⚡"):
            with st.spinner("Simulating..."):
                idx = df[df['Material_Name'] == sel_mat].index[0]
                N=1500; rng = np.random.default_rng()
                W_ops = rng.dirichlet(np.ones(3), N); W_oss = rng.dirichlet(np.ones(10), N)
                
                cloud_ops = np.exp(np.dot(W_ops, np.log(P_sc[idx]+1e-9)))
                cloud_oss = np.exp(np.dot(W_oss, np.log(S_sc[idx]+1e-9)))
                
                fig_mc = px.scatter(x=cloud_ops, y=cloud_oss, opacity=0.3, 
                                   title=f"Stability Cloud: {sel_mat}", labels={'x':'OPS','y':'OSS'},
                                   color_discrete_sequence=['#2563eb'])
                fig_mc.add_trace(go.Scatter(x=[df.loc[idx,'OPS']], y=[df.loc[idx,'OSS']], mode='markers', 
                                           marker=dict(color='red', size=15, symbol='star'), name='Current'))
                st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.error("CSV not found. Please upload 'MF_sustainability_rank.csv' to GitHub.")
