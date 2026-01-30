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
    
    :root { --primary: #1e3a8a; --bg: #f8fafc; }

    /* RESET GLOBALE */
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e3a8a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* SIDEBAR BIANCA */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* SCRITTA SETTINGS SEMPLICE */
    .settings-label {
        font-size: 20px; font-weight: 700; color: #1e3a8a;
        margin-bottom: 15px; padding-left: 5px;
    }

    /* HEADER SEZIONI BLU (TESTO BIANCO FORZATO) */
    .blue-section-header {
        background-color: #1e3a8a; padding: 10px 15px;
        border-radius: 8px; margin-top: 20px; margin-bottom: 10px;
    }
    .blue-section-header p {
        color: #ffffff !important; margin: 0 !important;
        font-weight: 700 !important; font-size: 15px !important;
    }

    /* ELEMENT CONTAINERS (INPUT, SELECT, SUMMARY): BIANCO + BORDO GRIGIO */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    .summary-box {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    
    /* TESTO BLU DENTRO I BOX BIANCHI */
    input, span, .summary-box p {
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
        font-weight: 600;
    }

    /* BOTTONI +/- */
    div[data-baseweb="input"] button {
        background-color: #f1f5f9 !important;
        color: #1e3a8a !important;
    }
    
    /* LABEL SIDEBAR */
    section[data-testid="stSidebar"] label { color: #1e3a8a !important; font-weight: 700; }

    /* CARD PRINCIPALI */
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white !important; 
        border-radius: 12px; border: 1px solid #e2e8f0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTORE DI CALCOLO ---

def generate_linear_scores(n_tiers):
    """Calcola i coefficienti (es. 4 tiers = 1.0, 0.75, 0.5, 0.25)"""
    step = 1.0 / n_tiers
    return [round(1.0 - (i * step), 2) for i in range(n_tiers)]

def assign_tiered_scores(df, col_name, n_tiers, thresholds):
    """Assegna score basati su soglie manuali in ordine ascendente"""
    scores = generate_linear_scores(n_tiers)
    # Partiamo dallo score peggiore (l'ultimo della lista)
    assigned = pd.Series(scores[-1], index=df.index, dtype=float)
    
    # Accoppiamo soglie (thresholds) e score (dal Tier 2 al Tier 1)
    # manual_thresholds[0] è il limite per lo score 1.0
    # manual_thresholds[1] è il limite per lo score 0.75, etc.
    # Applichiamo dalla soglia più bassa alla più alta per sovrascrivere correttamente
    
    thresh_data = []
    for i in range(len(thresholds)):
        thresh_data.append((thresholds[i], scores[i]))
    
    # Ordiniamo per valore di soglia crescente
    thresh_data.sort(key=lambda x: x[0])
    
    for val, sc in thresh_data:
        assigned[df[col_name] >= val] = sc
    return assigned

def calculate_ops(df, t_map, config, weights):
    """Formula Prodotto: P1^w1 * P2^w2 * P3^w3"""
    p1 = assign_tiered_scores(df, 'P1', config['P1'], t_map['P1'])
    p2 = assign_tiered_scores(df, 'P2', config['P2'], t_map['P2'])
    p3 = assign_tiered_scores(df, 'P3', config['P3'], t_map['P3'])
    
    w = np.array(weights)
    if w.sum() > 0: w = w / w.sum()
    
    return np.power(p1, w[0]) * np.power(p2, w[1]) * np.power(p3, w[2])

def get_pareto_mask(pts):
    """Calcola i materiali sulla frontiera di Pareto"""
    efficient = np.ones(pts.shape[0], dtype=bool)
    for i, c in enumerate(pts):
        if efficient[i]:
            efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))
    return efficient

# --- 3. CARICAMENTO DATI ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("MF_sustainability_rank.csv")
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except: return None

# --- 4. INTERFACCIA APP ---
st.title("Materials Intelligence Platform")

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.markdown('<p class="settings-label">Settings</p>', unsafe_allow_html=True)
    
    manual_t = {'P1': [], 'P2': [], 'P3': []}
    
    with st.sidebar:
        # SEZIONE 1: TIERS & THRESHOLDS
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        
        # P1 TEMPERATURE
        st.markdown("**P1: Temperature (K)**")
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Tier {i+1} Limit (Score {sc_t[i]})", value=350.0 + (100/(i+1)), min_value=350.0, key=f"p1_{i}")
            manual_t['P1'].append(val)
        
        # P2 MAGNETIZATION
        st.markdown("---")
        st.markdown("**P2: Magnetization (T)**")
        sf_m = st.selectbox("Subcategories (P2)", [2, 3, 4, 5], index=1)
        sc_m = generate_linear_scores(sf_m)
        for i in range(sf_m - 1):
            val = st.number_input(f"Tier {i+1} Limit (Score {sc_m[i]})", value=0.4 + (0.2/(i+1)), min_value=0.4, key=f"p2_{i}")
            manual_t['P2'].append(val)

        # P3 COERCIVITY
        st.markdown("---")
        st.markdown("**P3: Coercivity (T)**")
        sf_c = st.selectbox("Subcategories (P3)", [2, 3, 4, 5], index=3)
        sc_c = generate_linear_scores(sf_c)
        for i in range(sf_c - 1):
            val = st.number_input(f"Tier {i+1} Limit (Score {sc_c[i]})", value=0.4 + (0.2/(i+1)), min_value=0.4, key=f"p3_{i}")
            manual_t['P3'].append(val)

        # SEZIONE 2: COEFFICIENTS
        st.markdown('<div class="blue-section-header"><p>2. Performance Coefficients</p></div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = round(max(0.0, 1.0 - (w_p1 + w_p2)), 2)
        
        st.markdown(f"""
        <div class="summary-box" style="padding: 10px; text-align: center; margin-top: 10px;">
            <p style="margin:0; font-size: 14px;">Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CALCOLI CORE ---
    config = {'P1': sf_t, 'P2': sf_m, 'P3': sf_c}
    weights = [w_p1, w_p2, w_p3]
    
    # Calcolo OPS (Performance)
    df['OPS'] = calculate_ops(df, manual_t, config, weights)
    
    # Calcolo OSS (Sustainability - Media Geometrica S1-S10)
    s_cols = [f'S{i}' for i in range(1, 11)]
    if all(c in df.columns for c in s_cols):
        s_data = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
        df['OSS'] = np.exp(np.mean(np.log(np.clip(s_data, 1e-3, 1.0)), axis=1))
    else:
        df['OSS'] = 0.5

    # --- MAIN CONTENT (TABS) ---
    t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "🔬 Stability Analysis"])

    # TAB 1: RANKING
    with t1:
        colA, colB = st.columns([2, 1])
        with colA:
            pts = df[['OPS', 'OSS']].to_numpy()
            efficient = get_pareto_mask(pts)
            df['Status'] = np.where(efficient, 'Optimal Choice', 'Standard')
            
            fig = px.scatter(df, x='OPS', y='OSS', color='Status', hover_name='Material_Name',
                             color_discrete_map={'Optimal Choice': '#1e3a8a', 'Standard': '#cbd5e1'})
            fig.update_layout(template="plotly_white", xaxis_title="OPS (Performance)", yaxis_title="OSS (Sustainability)")
            st.plotly_chart(fig, use_container_width=True)
        
        with colB:
            st.markdown("**Top Pareto Materials**")
            st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], 
                         use_container_width=True, height=500)

    # TAB 2: SCALABILITY
    with t2:
        if 'Pmax_t_per_yr' in df.columns and 'Plong_t' in df.columns:
            fig_sc = px.scatter(df, x='Plong_t', y='Pmax_t_per_yr', color='OSS', log_x=True, log_y=True,
                                hover_name='Material_Name', color_continuous_scale="Viridis",
                                labels={'Plong_t': 'Reserves (t)', 'Pmax_t_per_yr': 'Production (t/yr)'})
            fig_sc.update_layout(template="plotly_white")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("Missing Supply Data (Pmax/Plong) in CSV.")

    # TAB 3: STABILITY (MONTE CARLO)
    with t3:
        st.markdown("### Robustness Simulation")
        opts = df[efficient]['Material_Name'].unique()
        if len(opts) > 0:
            sel = st.selectbox("Select a Material to test:", opts)
            if st.button("Run Simulation ⚡"):
                idx = df[df['Material_Name'] == sel].index[0]
                rng = np.random.default_rng()
                
                # Simulazione variazione pesi (Dirichlet)
                W_ops_sim = rng.dirichlet(np.array(weights)*50 + 1, 1000)
                
                # Recupero punteggi base Tier per il materiale
                p1_s = assign_tiered_scores(df.iloc[[idx]], 'P1', sf_t, manual_t['P1']).values[0]
                p2_s = assign_tiered_scores(df.iloc[[idx]], 'P2', sf_m, manual_t['P2']).values[0]
                p3_s = assign_tiered_scores(df.iloc[[idx]], 'P3', sf_c, manual_t['P3']).values[0]
                s_vec = np.array([p1_s, p2_s, p3_s])
                
                # Nuvola OPS
                c_ops = np.exp(np.dot(W_ops_sim, np.log(s_vec + 1e-9)))
                
                # Nuvola OSS
                W_oss_sim = rng.dirichlet(np.ones(10)*20, 1000)
                s_oss_vec = df.loc[idx, s_cols].to_numpy(dtype=float)
                c_oss = np.exp(np.dot(W_oss_sim, np.log(np.clip(s_oss_vec, 1e-3, 1.0))))
                
                fig_mc = px.scatter(x=c_ops, y=c_oss, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                fig_mc.add_trace(go.Scatter(x=[df.loc[idx,'OPS']], y=[df.loc[idx,'OSS']], mode='markers',
                                          marker=dict(color='red', size=12, symbol='star'), name='Current Point'))
                fig_mc.update_layout(template="plotly_white", xaxis_title="OPS Stability", yaxis_title="OSS Stability")
                st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("No optimal materials to simulate.")

else:
    st.error("Please ensure 'MF_sustainability_rank.csv' is available.")
