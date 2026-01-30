import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAZIONE E STILE ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root { --primary: #1e3a8a; --bg: #f8fafc; }

    /* Reset Globale */
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e3a8a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* Sidebar Bianca */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Settings Label Blu Semplice */
    .settings-title {
        font-size: 20px; font-weight: 700; color: #1e3a8a;
        margin-bottom: 15px; padding-left: 5px;
    }

    /* Box Titoli Sezioni (Blu con testo Bianco Forzato) */
    .blue-section-header {
        background-color: #1e3a8a; padding: 10px 15px;
        border-radius: 8px; margin-top: 20px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .blue-section-header p {
        color: #ffffff !important; margin: 0 !important;
        font-weight: 700 !important; font-size: 15px !important;
    }

    /* Element Containers (Input, Select, Summary): Bianco con Bordo Grigio */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    .custom-summary-box {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #1e3a8a !important;
    }
    
    /* Testo Blu dentro i box bianchi */
    input, span, .custom-summary-box p {
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
        font-weight: 600;
    }

    /* Pulsanti +/- */
    div[data-baseweb="input"] button {
        background-color: #f1f5f9 !important;
        color: #1e3a8a !important;
    }
    
    /* Label Sidebar */
    section[data-testid="stSidebar"] label {
        color: #1e3a8a !important; font-weight: 700;
    }

    /* Card Principali */
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white !important; 
        border-radius: 12px; border: 1px solid #e2e8f0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTORE DI CALCOLO ---

def generate_linear_scores(n_tiers):
    """Genera punteggi crescenti: [1/N, 2/N, ..., 1.0]"""
    return [round((i + 1) / n_tiers, 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, n_tiers, thresholds):
    """Assegna score basati su soglie manuali in ordine ASCENDENTE."""
    scores = generate_linear_scores(n_tiers)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    
    # Applichiamo dalla soglia più bassa alla più alta per sovrascrivere verso l'alto
    for i in range(len(thresholds)):
        val = thresholds[i]
        sc = scores[i+1]
        assigned[df[col_name] >= val] = sc
    return assigned

# --- 3. CARICAMENTO DATI ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("MF_sustainability_rank.csv")
        for col in ['P1', 'P2', 'P3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception:
        return None

# --- 4. INTERFACCIA APP ---
st.title("Materials Intelligence Platform")

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.markdown('<p class="settings-title">Settings</p>', unsafe_allow_html=True)
    
    manual_thresholds = {'P1': [], 'P2': [], 'P3': []}
    is_valid = True
    
    with st.sidebar:
        # SEZIONE 1: TIERS & THRESHOLDS
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        
        # P1: TEMP (Step impostato a 1.0 per cambiare le unità)
        st.markdown("**P1: Temperature (K)**")
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Threshold for Score {sc_t[i+1]} (P1)", 
                                  value=350.0 + (i*50), 
                                  min_value=350.0, 
                                  step=1.0, 
                                  key=f"p1_{i}")
            manual_thresholds['P1'].append(val)
        if any(manual_thresholds['P1'][i] >= manual_thresholds['P1'][i+1] for i in range(len(manual_thresholds['P1'])-1)):
            st.error("Error: P1 thresholds must be strictly ascending!")
            is_valid = False

        # P2: MAG
        st.markdown("---")
        st.markdown("**P2: Magnetization (T)**")
        sf_m = st.selectbox("Subcategories (P2)", [2, 3, 4, 5], index=1)
        sc_m = generate_linear_scores(sf_m)
        for i in range(sf_m - 1):
            val = st.number_input(f"Threshold for Score {sc_m[i+1]} (P2)", value=0.4 + (i*0.2), min_value=0.4, key=f"p2_{i}")
            manual_thresholds['P2'].append(val)
        if any(manual_thresholds['P2'][i] >= manual_thresholds['P2'][i+1] for i in range(len(manual_thresholds['P2'])-1)):
            st.error("Error: P2 thresholds must be strictly ascending!")
            is_valid = False

        # P3: COERC
        st.markdown("---")
        st.markdown("**P3: Coercivity (T)**")
        sf_c = st.selectbox("Subcategories (P3)", [2, 3, 4, 5], index=3)
        sc_c = generate_linear_scores(sf_c)
        for i in range(sf_c - 1):
            val = st.number_input(f"Threshold for Score {sc_c[i+1]} (P3)", value=0.4 + (i*0.2), min_value=0.4, key=f"p3_{i}")
            manual_thresholds['P3'].append(val)
        if any(manual_thresholds['P3'][i] >= manual_thresholds['P3'][i+1] for i in range(len(manual_thresholds['P3'])-1)):
            st.error("Error: P3 thresholds must be strictly ascending!")
            is_valid = False

        # SEZIONE 2: COEFFICIENTS
        st.markdown('<div class="blue-section-header"><p>2. Performance Coefficients</p></div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = round(1.0 - w_p1, 2)
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, rem, min(0.33, rem))
        w_p3 = round(max(0.0, 1.0 - (w_p1 + w_p2)), 2)
        
        st.markdown(f"""
        <div class="custom-summary-box" style="padding: 10px; text-align: center; margin-top: 10px;">
            <p style="margin:0; font-size: 14px;">Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CALCOLO E VISUALIZZAZIONE ---
    if is_valid:
        # Calcolo Score Performance (OPS) = P1^w1 * P2^w2 * P3^w3
        p1_s = assign_tiered_scores(df, 'P1', sf_t, manual_thresholds['P1'])
        p2_s = assign_tiered_scores(df, 'P2', sf_m, manual_thresholds['P2'])
        p3_s = assign_tiered_scores(df, 'P3', sf_c, manual_thresholds['P3'])
        
        df['OPS'] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)
        
        # Calcolo Score Sostenibilità (OSS)
        s_cols = [f'S{i}' for i in range(1, 11)]
        if all(c in df.columns for c in s_cols):
            s_data = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
            df['OSS'] = np.exp(np.mean(np.log(np.clip(s_data, 1e-3, 1.0)), axis=1))
        else:
            df['OSS'] = 0.5

        # --- TABS ---
        t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "🔬 Stability Analysis"])

        with t1:
            colA, colB = st.columns([2, 1])
            with colA:
                pts = df[['OPS', 'OSS']].to_numpy()
                efficient = np.ones(pts.shape[0], dtype=bool)
                for i, c in enumerate(pts):
                    if efficient[i]:
                        efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))
                df['Status'] = np.where(efficient, 'Optimal Choice', 'Standard')
                
                fig = px.scatter(df, x='OPS', y='OSS', color='Status', hover_name='Material_Name',
                                 color_discrete_map={'Optimal Choice': '#1e3a8a', 'Standard': '#cbd5e1'})
                fig.update_layout(template="plotly_white", xaxis_title="OPS (Performance)", yaxis_title="OSS (Sustainability)")
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Pareto Materials**")
                st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], 
                             use_container_width=True, height=500)

        with t2:
            if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
                fig_sc = px.scatter(df, x='Plong_t', y='Pmax_t_per_yr', color='OSS', log_x=True, log_y=True,
                                    hover_name='Material_Name', color_continuous_scale="Viridis",
                                    labels={'Plong_t': 'Reserves (t)', 'Pmax_t_per_yr': 'Production (t/yr)'})
                fig_sc.update_layout(template="plotly_white")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.warning("Missing Supply Data (Pmax/Plong) in CSV.")

        with t3:
            st.markdown("### Robustness Simulation")
            opts = df[efficient]['Material_Name'].unique()
            if len(opts) > 0:
                sel = st.selectbox("Select a Material to test:", opts)
                if st.button("Run Simulation ⚡"):
                    idx = df[df['Material_Name'] == sel].index[0]
                    rng = np.random.default_rng()
                    W_sim = rng.dirichlet(np.array([w_p1, w_p2, w_p3])*50 + 1, 1000)
                    s_vec = np.array([p1_s[idx], p2_s[idx], p3_s[idx]])
                    c_ops = np.exp(np.dot(W_sim, np.log(s_vec + 1e-9)))
                    
                    fig_mc = px.scatter(x=c_ops, y=[df.loc[idx, 'OSS']]*1000, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                    fig_mc.add_trace(go.Scatter(x=[df.loc[idx, 'OPS']], y=[df.loc[idx, 'OSS']], mode='markers',
                                              marker=dict(color='red', size=12, symbol='star'), name='Current Point'))
                    fig_mc.update_layout(template="plotly_white", xaxis_title="OPS Stability", yaxis_title="OSS Stability")
                    st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.warning("Please correct the threshold order in the sidebar to proceed.")
else:
    st.error("Ensure 'MF_sustainability_rank.csv' is in the working directory.")
