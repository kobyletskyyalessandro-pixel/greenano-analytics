import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAZIONE PAGINA E STILE (MATCHING EXACT WEBSITE STYLE) ---
st.set_page_config(
    page_title="GreeNano Analytics",
    page_icon="🔬",
    layout="wide"
)

# Inseriamo il CSS esatto del tuo sito HTML
st.markdown("""
    <style>
    /* Import Font Inter come nel sito */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Variabili Colore dal tuo CSS originale */
    :root {
        --primary: #1e3a8a;       /* Midnight Blue */
        --secondary: #2563eb;     /* Royal Blue */
        --accent: #22c55e;        /* Green Accent */
        --bg: #f8fafc;           /* Light Gray Background */
        --text: #0f172a;         /* Dark Slate Text */
        --muted: #64748b;        /* Muted Text */
    }

    /* Override Generale Streamlit */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }
    
    /* Sfondo App */
    .stApp {
        background-color: var(--bg);
    }

    /* Titoli */
    h1, h2, h3 {
        color: var(--primary) !important;
        font-weight: 700;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Bottoni (Stile esatto del sito) */
    div.stButton > button:first-child {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 22px;
        font-size: 15px;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        color: white;
    }

    /* Card Effect per i grafici */
    div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        border-radius: 12px;
        padding: 10px;
        /* box-shadow: 0 4px 6px rgba(0,0,0,0.05); */ /* Opzionale: ombra leggera */
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTORE DI CALCOLO ---

def weighted_geometric_mean(scores_matrix, weights, eps=1e-12):
    S = np.asarray(scores_matrix, dtype=float)
    S = np.clip(S, eps, 1.0)
    w = np.asarray(weights, dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    return np.exp(np.sum(w * np.log(S), axis=1))

def pareto_front(points):
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]: continue
        dominates_i = np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)
        if np.any(dominates_i): 
            efficient[i] = False
        else:
            dominated_by_i = np.all(P[i] >= P, axis=1) & np.any(P[i] > P, axis=1)
            efficient[dominated_by_i] = False
    return efficient

# --- 3. CARICAMENTO DATI ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Assicurati che il file su GitHub si chiami esattamente così
        df = pd.read_csv("MF_sustainability_rank.csv")
        return df
    except FileNotFoundError:
        return None

# --- 4. INTERFACCIA UTENTE ---

st.title("Materials Intelligence Platform")
st.markdown("""
<div style="color: #64748b; font-size: 16px; margin-bottom: 25px;">
    Advanced Analytics Module. Adjust weights to explore trade-offs between Performance and Sustainability in real-time.
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Analysis Parameters")
    
    st.sidebar.subheader("Performance Weights")
    st.sidebar.markdown("<span style='color:#64748b; font-size:12px'>Sum must be 1.0</span>", unsafe_allow_html=True)
    
    w_p1 = st.sidebar.slider("Weight P1 (Magnetization)", 0.0, 1.0, 0.33)
    remaining_after_p1 = 1.0 - w_p1
    w_p2 = st.sidebar.slider("Weight P2 (Anisotropy)", 0.0, max(0.0, remaining_after_p1), min(0.33, remaining_after_p1))
    w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
    
    # Info Box stile "Sito"
    st.sidebar.markdown(f"""
    <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-top: 10px;">
        <strong style="color: #1e3a8a">Current Weights:</strong><br>
        <span style="color: #0f172a">P1: {w_p1:.2f} | P2: {w_p2:.2f} | P3: {w_p3:.2f}</span><br>
        <hr style="margin: 8px 0; border-color: #cbd5e1">
        <strong>Total: {w_p1+w_p2+w_p3:.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.subheader("Sustainability Balance")
    w_oss = st.sidebar.slider("Global Sustainability Weight", 0.0, 1.0, 0.5)

    # --- CALCOLI ---
    p_weights = np.array([w_p1, w_p2, w_p3])
    s_weights = np.ones(10) / 10
    
    p_cols = ["P1", "P2", "P3"]
    s_cols = [f"S{i}" for i in range(1, 11)]
    
    missing_cols = [c for c in p_cols + s_cols if c not in df.columns]
    
    if not missing_cols:
        P_scores = df[p_cols].apply(pd.to_numeric, errors='coerce').fillna(0.3).to_numpy()
        S_mat = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.3).to_numpy()
        
        df['OPS'] = weighted_geometric_mean(P_scores, p_weights)
        df['OSS'] = weighted_geometric_mean(S_mat, s_weights)

        # --- GRAFICI ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Pareto Frontier")
            mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
            df['Type'] = np.where(mask, 'Pareto Efficient', 'Standard')
            
            # Colori: Blu Scuro (Primary) per i migliori, Azzurro (Secondary) per gli altri
            fig_pareto = px.scatter(
                df, x='OPS', y='OSS', color='Type', 
                hover_name='Material_Name',
                hover_data=['Chemical_Formula'] if 'Chemical_Formula' in df.columns else None,
                color_discrete_map={'Pareto Efficient': '#1e3a8a', 'Standard': '#93c5fd'},
                title=""
            )
            # Aggiungo linea Pareto
            pareto_points = df[mask].sort_values(by="OPS")
            fig_pareto.add_trace(go.Scatter(
                x=pareto_points['OPS'], y=pareto_points['OSS'],
                mode='lines', name='Frontier', line=dict(color='#1e3a8a', width=2, dash='dash')
            ))
            fig_pareto.update_layout(
                template="plotly_white",
                xaxis_title="OPS (Performance Score)",
                yaxis_title="OSS (Sustainability Score)",
                font=dict(family="Inter, sans-serif")
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

        with col2:
            st.subheader("Industrial Scalability")
            if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
                fig_scale = px.scatter(
                    df, x='Plong_t', y='Pmax_t_per_yr', color='OSS',
                    log_x=True, log_y=True, hover_name='Material_Name',
                    color_continuous_scale="Blues", # Scala Blu
                    title=""
                )
                fig_scale.update_layout(
                    template="plotly_white",
                    xaxis_title="Reserves (Plong_t)",
                    yaxis_title="Max Prod (t/yr)",
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig_scale, use_container_width=True)
            else:
                st.info("Scalability data columns missing.")

        # --- DATA TABLE ---
        st.subheader("Detailed Results")
        st.dataframe(
            df[['Material_Name', 'OPS', 'OSS', 'Chemical_Formula'] if 'Chemical_Formula' in df.columns else ['Material_Name', 'OPS', 'OSS']], 
            use_container_width=True
        )

        # --- MONTE CARLO ---
        st.divider()
        st.header("🔬 Uncertainty Analysis")
        st.markdown("Run a Monte Carlo simulation to test the stability of a material's score.")
        
        sorted_materials = df.sort_values(by="OPS", ascending=False)['Material_Name'].unique()
        col_sel, col_btn = st.columns([3, 1])
        
        with col_sel:
            selected_material_name = st.selectbox("Select Material:", sorted_materials)
        
        with col_btn:
            st.write("") # Spacer
            st.write("") # Spacer
            run_sim = st.button("Run Simulation ⚡")

        if run_sim:
            with st.spinner("Simulating scenarios..."):
                mat_idx = df[df['Material_Name'] == selected_material_name].index[0]
                n_samples = 2000
                rng = np.random.default_rng()
                
                W_ops = rng.dirichlet(np.ones(3), size=n_samples)
                W_oss = rng.dirichlet(np.ones(10), size=n_samples)
                
                p_vec = P_scores[mat_idx]
                s_vec = S_mat[mat_idx]
                
                ops_cloud = np.exp(np.dot(W_ops, np.log(p_vec + 1e-12)))
                oss_cloud = np.exp(np.dot(W_oss, np.log(s_vec + 1e-12)))

                cloud_df = pd.DataFrame({'OPS': ops_cloud, 'OSS': oss_cloud})
                
                fig_cloud = px.scatter(
                    cloud_df, x="OPS", y="OSS", 
                    opacity=0.3, color_discrete_sequence=['#2563eb'] # Royal Blue
                )
                
                fig_cloud.add_trace(go.Scatter(
                    x=[df.loc[mat_idx, 'OPS']], y=[df.loc[mat_idx, 'OSS']],
                    mode='markers', name='Current Selection',
                    marker=dict(color='#dc2626', size=15, symbol='star') # Rosso per contrasto
                ))

                fig_cloud.update_layout(
                    template="plotly_white", 
                    xaxis_range=[0,1], yaxis_range=[0,1],
                    title=f"Stability Cloud: {selected_material_name}",
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig_cloud, use_container_width=True)

    else:
        st.error(f"Missing columns: {missing_cols}")
else:
    st.warning("⚠️ CSV file not found. Please upload 'MF_sustainability_rank.csv' to GitHub.")
