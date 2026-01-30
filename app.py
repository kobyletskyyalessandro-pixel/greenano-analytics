import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="GreeNano Material Explorer", layout="wide")

# --- MOTORE DI CALCOLO (FUNZIONI) ---

def weighted_geometric_mean(scores_matrix, weights, eps=1e-12):
    """Calcola la media geometrica ponderata (OPS/OSS)"""
    S = np.asarray(scores_matrix, dtype=float)
    S = np.clip(S, eps, 1.0)
    w = np.asarray(weights, dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    return np.exp(np.sum(w * np.log(S), axis=1))

def pareto_front(points):
    """Trova i punti sulla frontiera di Pareto"""
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

# --- INTERFACCIA UTENTE ---

st.title("🍀 GreeNano Material Explorer")
st.markdown("Explore material sustainability and performance in real-time.")

# Caricamento Dati
@st.cache_data
def load_data():
    try:
        # Assicurati che il nome del file sia corretto
        df = pd.read_csv("MF_sustainability_rank (4).csv")
        return df
    except FileNotFoundError:
        return None

# --- BLOCCO PRINCIPALE ---
try:
    df = load_data()

    if df is not None:
        # --- SIDEBAR: CONTROLLI UTENTE ---
        st.sidebar.header("Analysis Parameters")
        
        # 1. PESI PERFORMANCE (Logica Somma = 1)
        st.sidebar.subheader("Performance Weights (Sum must be 1.0)")
        
        w_p1 = st.sidebar.slider("Weight P1 (e.g. Conductivity)", 0.0, 1.0, 0.33)
        remaining_after_p1 = 1.0 - w_p1
        w_p2 = st.sidebar.slider("Weight P2 (e.g. Cost)", 0.0, max(0.0, remaining_after_p1), min(0.33, remaining_after_p1))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        st.sidebar.info(f"🔹 P1: {w_p1:.2f}\n🔹 P2: {w_p2:.2f}\n🔹 P3: {w_p3:.2f}\n\n**Total: {w_p1+w_p2+w_p3:.2f}**")

        st.sidebar.subheader("Sustainability Balance")
        w_oss = st.sidebar.slider("S1-S10 Balance", 0.0, 1.0, 0.5)

        # --- CALCOLO PUNTEGGI ---
        p_weights = np.array([w_p1, w_p2, w_p3])
        s_weights = np.ones(10) / 10
        
        p_cols = ["P1", "P2", "P3"]
        s_cols = [f"S{i}" for i in range(1, 11)]
        
        # Gestione colonne mancanti
        missing_cols = [c for c in p_cols + s_cols if c not in df.columns]
        if not missing_cols:
            P_scores = df[p_cols].apply(pd.to_numeric, errors='coerce').fillna(0.3).to_numpy()
            S_mat = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.3).to_numpy()
            
            df['OPS'] = weighted_geometric_mean(P_scores, p_weights)
            df['OSS'] = weighted_geometric_mean(S_mat, s_weights)

            # --- GRAFICI PRINCIPALI (Pareto & Scalability) ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Pareto Frontier")
                mask = pareto_front(df[['OPS', 'OSS']].to_numpy())
                df['Pareto Status'] = np.where(mask, 'Frontier (Best)', 'Others')
                
                fig_pareto = px.scatter(
                    df, x='OPS', y='OSS', color='Pareto Status', 
                    hover_name='Material_Name',
                    hover_data=['Chemical_Formula'] if 'Chemical_Formula' in df.columns else None,
                    color_discrete_map={'Frontier (Best)': '#EF553B', 'Others': '#636EFA'},
                    title="Performance vs Sustainability"
                )
                pareto_points = df[mask].sort_values(by="OPS")
                fig_pareto.add_trace(go.Scatter(
                    x=pareto_points['OPS'], y=pareto_points['OSS'],
                    mode='lines', name='Pareto Line', line=dict(color='rgba(239, 85, 59, 0.3)', width=2)
                ))
                st.plotly_chart(fig_pareto, use_container_width=True)

            with col2:
                st.subheader("Scalability")
                if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
                    fig_scale = px.scatter(
                        df, x='Plong_t', y='Pmax_t_per_yr', color='OSS',
                        log_x=True, log_y=True, hover_name='Material_Name',
                        color_continuous_scale="Viridis",
                        title="Industrial Scalability"
                    )
                    st.plotly_chart(fig_scale, use_container_width=True)
                else:
                    st.info("Columns 'Pmax_t_per_yr' and 'Plong_t' not found.")

            st.dataframe(df[['Material_Name', 'OPS', 'OSS']])

            # --- NUOVA SEZIONE: INCERTEZZA (Codice 2) ---
            st.divider()
            st.header("🔬 Uncertainty Analysis (Monte Carlo)")
            st.markdown("Select a material to simulate how its score changes if weights vary slightly.")

            # Ordiniamo i materiali per OPS
            sorted_materials = df.sort_values(by="OPS", ascending=False)['Material_Name'].unique()
            selected_material_name = st.selectbox("Choose Material to Analyze:", sorted_materials)

            if st.button(f"Run Simulation for {selected_material_name}"):
                with st.spinner("Simulating 2,000 scenarios..."):
                    # Trova l'indice del materiale
                    mat_idx = df[df['Material_Name'] == selected_material_name].index[0]
                    
                    # Generazione campioni casuali
                    n_samples = 2000
                    rng = np.random.default_rng()
                    W_ops = rng.dirichlet(np.ones(3), size=n_samples)
                    W_oss = rng.dirichlet(np.ones(10), size=n_samples)

                    # Dati per la simulazione
                    p_vec = P_scores[mat_idx]
                    s_vec = S_mat[mat_idx]
                    
                    # Calcolo nuvola punti
                    ops_cloud = np.exp(np.dot(W_ops, np.log(p_vec + 1e-12)))
                    oss_cloud = np.exp(np.dot(W_oss, np.log(s_vec + 1e-12)))

                    # Plotting
                    cloud_df = pd.DataFrame({'OPS': ops_cloud, 'OSS': oss_cloud})
                    
                    fig_cloud = px.scatter(
                        cloud_df, x="OPS", y="OSS", 
                        title=f"Stability Cloud for {selected_material_name}",
                        opacity=0.2, render_mode='webgl'
                    )
                    
                    # Aggiungi punto attuale
                    current_ops = df.loc[mat_idx, 'OPS']
                    current_oss = df.loc[mat_idx, 'OSS']
                    
                    fig_cloud.add_trace(go.Scatter(
                        x=[current_ops], y=[current_oss],
                        mode='markers', name='Current Selection',
                        marker=dict(color='red', size=12, symbol='star')
                    ))

                    fig_cloud.update_layout(
                        xaxis_title="OPS (Performance)", yaxis_title="OSS (Sustainability)",
                        template="plotly_white", xaxis_range=[0, 1], yaxis_range=[0, 1]
                    )
                    
                    
                    st.plotly_chart(fig_cloud, use_container_width=True)
                    st.info(f"Mean OPS: {ops_cloud.mean():.3f} (±{ops_cloud.std():.3f}) | Mean OSS: {oss_cloud.mean():.3f} (±{oss_cloud.std():.3f})")

        else:
            st.error(f"Missing columns: {missing_cols}")
    else:
        st.error("File 'MF_sustainability_rank (4).csv' not found. Please put it in the same folder.")

except Exception as e:
    st.error(f"An error occurred: {e}")