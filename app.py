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
    
    div.stButton > button:first-child { 
        background-color: var(--primary); 
        color: white; 
        border-radius: 8px; 
        border: none; 
        padding: 12px 24px; 
        font-weight: 600; 
        transition: all 0.2s;
    }
    div.stButton > button:hover { 
        background-color: var(--secondary); 
        transform: translateY(-2px); 
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white; 
        border-radius: 12px; 
        padding: 20px; 
        border: 1px solid #e2e8f0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTORE DI CALCOLO ---
def weighted_geometric_mean(S, w, eps=1e-12):
    """Calcola score geometrico pesato (OPS/OSS)"""
    S = np.clip(np.asarray(S, dtype=float), eps, 1.0)
    w = np.asarray(w, dtype=float)
    if w.sum() > 0: w = w / w.sum()
    return np.exp(np.sum(w * np.log(S), axis=1))

def pareto_front(points):
    """Calcola la frontiera di Pareto"""
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
        # NOME FILE CONFERMATO DAL TUO COLLAB (Cella 74)
        df = pd.read_csv("MF_sustainability_rank.csv")
        return df
    except FileNotFoundError:
        return None

# --- 4. INTERFACCIA UTENTE ---
st.title("Materials Intelligence Platform")
st.markdown("""
<div style="color: #64748b; font-size: 16px; margin-bottom: 25px;">
    <b>Scientific Dashboard:</b> Real-time weighting, Supply Chain criticality analysis, and Pareto optimization.
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Model Parameters")
    
    st.sidebar.subheader("Performance Weights")
    w_p1 = st.sidebar.slider("Weight P1 (Magnetization)", 0.0, 1.0, 0.33)
    rem = 1.0 - w_p1
    w_p2 = st.sidebar.slider("Weight P2 (Anisotropy)", 0.0, max(0.0, rem), min(0.33, rem))
    w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
    
    st.sidebar.markdown(f"""
    <div style="background: #eff6ff; border: 1px solid #dbeafe; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
        <strong style="color: #1e3a8a">Current Split:</strong><br>
        🔹 P1: {w_p1:.2f}<br>🔹 P2: {w_p2:.2f}<br>🔹 P3: {w_p3:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.subheader("Sustainability Balance")
    w_oss = st.sidebar.slider("Sustainability Priority", 0.0, 1.0, 0.5)

    # --- CALCOLO REAL-TIME ---
    cols_p = ["P1", "P2", "P3"]
    cols_s = [f"S{i}" for i in range(1, 11)]
    
    if all(c in df.columns for c in cols_p + cols_s):
        P_sc = df[cols_p].fillna(0.3).to_numpy()
        S_sc = df[cols_s].fillna(0.3).to_numpy()
        
        df['OPS'] = weighted_geometric_mean(P_sc, [w_p1, w_p2, w_p3])
        df['OSS'] = weighted_geometric_mean(S_sc, np.ones(10)/10)

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🏆 Pareto Ranking", "🏭 Supply Chain (Paper)", "🔬 Stability Analysis"])

        # TAB 1: RANKING
        with tab1:
            colA, colB = st.columns([2, 1])
            with colA:
                st.subheader("Performance vs Sustainability Frontier")
                mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
                df['Status'] = np.where(mask, 'Pareto Efficient', 'Dominated')
                
                fig_pareto = px.scatter(
                    df, x='OPS', y='OSS', color='Status',
                    hover_name='Material_Name', hover_data=['Chemical_Formula'],
                    color_discrete_map={'Pareto Efficient': '#1e3a8a', 'Dominated': '#93c5fd'},
                )
                fig_pareto.update_layout(template="plotly_white", xaxis_title="OPS (Performance)", yaxis_title="OSS (Sustainability)")
                st.plotly_chart(fig_pareto, use_container_width=True)
            
            with colB:
                st.subheader("Top Performers")
                st.dataframe(df[mask].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], use_container_width=True, height=400)

        # TAB 2: GRAFICI PAPER
        with tab2:
            st.markdown("### 🌍 Criticality & Forecast Analysis")
            
            # Check colonne necessarie per FiguresPaper
            if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
                col1, col2 = st.columns(2)
                
                # 1. SCALABILITY MAP
                with col1:
                    st.subheader("1. Scalability Map (Weakest-Link)")
                    st.caption("Do materials have enough Reserves (x) and Production Capacity (y)?")
                    fig_scale = px.scatter(
                        df, x='Plong_t', y='Pmax_t_per_yr', color='OSS',
                        log_x=True, log_y=True, hover_name='Material_Name',
                        color_continuous_scale="Viridis",
                        labels={'Plong_t': 'Reserves (tons)', 'Pmax_t_per_yr': 'Max Yearly Prod (t/yr)', 'OSS': 'Sust. Score'}
                    )
                    fig_scale.update_layout(template="plotly_white")
                    st.plotly_chart(fig_scale, use_container_width=True)

                # 2. CORRELATION / HIGH-RIGHT SCORE
                with col2:
                    st.subheader("2. Sustainability vs Availability")
                    st.caption("Correlation between Supply Security (Log Pmax + Log Plong) and Sustainability.")
                    
                    valid_df = df[(df['Pmax_t_per_yr'] > 0) & (df['Plong_t'] > 0)].copy()
                    valid_df['H_Score'] = np.log10(valid_df['Pmax_t_per_yr']) + np.log10(valid_df['Plong_t'])
                    
                    fig_corr = px.scatter(
                        valid_df, x='H_Score', y='OSS', hover_name='Material_Name',
                        trendline="ols", trendline_color_override="#dc2626",
                        labels={'H_Score': 'Supply Index (Log Pmax + Log Plong)', 'OSS': 'Sustainability Score'}
                    )
                    fig_corr.update_layout(template="plotly_white")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    corr_val = valid_df['H_Score'].corr(valid_df['OSS'])
                    st.info(f"📊 **Correlation:** {corr_val:.3f}")

            else:
                st.error("⚠️ Supply Data Missing: CSV must have 'Pmax_t_per_yr' and 'Plong_t'.")

        # TAB 3: MONTE CARLO
        with tab3:
            st.header("🔬 Uncertainty & Stability")
            sel_mat = st.selectbox("Select Material:", df.sort_values(by='OPS', ascending=False)['Material_Name'].unique())
            
            if st.button("Run Simulation ⚡"):
                with st.spinner("Simulating..."):
                    idx = df[df['Material_Name'] == sel_mat].index[0]
                    N = 2000
                    rng = np.random.default_rng()
                    W_ops_sim = rng.dirichlet(np.ones(3), N)
                    W_oss_sim = rng.dirichlet(np.ones(10), N)
                    
                    cloud_ops = np.exp(np.dot(W_ops_sim, np.log(P_sc[idx] + 1e-9)))
                    cloud_oss = np.exp(np.dot(W_oss_sim, np.log(S_sc[idx] + 1e-9)))
                    
                    fig_mc = px.scatter(
                        x=cloud_ops, y=cloud_oss, opacity=0.2,
                        title=f"Stability Cloud for {sel_mat}",
                        labels={'x': 'OPS', 'y': 'OSS'},
                        color_discrete_sequence=['#2563eb']
                    )
                    fig_mc.add_trace(go.Scatter(
                        x=[df.loc[idx, 'OPS']], y=[df.loc[idx, 'OSS']],
                        mode='markers', name='Current Weighting',
                        marker=dict(color='#dc2626', size=15, symbol='star')
                    ))
                    fig_mc.update_layout(template="plotly_white", xaxis_range=[0,1], yaxis_range=[0,1])
                    st.plotly_chart(fig_mc, use_container_width=True)

    else:
        st.error("⚠️ Critical columns missing in CSV.")

else:
    st.warning("⚠️ Data file not found. Please upload 'MF_sustainability_rank.csv' to GitHub.")
