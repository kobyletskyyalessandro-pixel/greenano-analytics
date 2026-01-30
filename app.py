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
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e3a8a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .sidebar-title { font-size: 20px; font-weight: 700; color: #1e3a8a; margin-bottom: 15px; }
    .blue-section-header { background-color: #1e3a8a; padding: 10px 15px; border-radius: 8px; margin-top: 20px; margin-bottom: 10px; }
    .blue-section-header p { color: #ffffff !important; margin: 0 !important; font-weight: 700 !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"], .custom-summary-box {
        background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    }
    input, span, .custom-summary-box p { color: #1e3a8a !important; font-weight: 600; }
    div[data-baseweb="input"] button { background-color: #f1f5f9 !important; color: #1e3a8a !important; }
    section[data-testid="stSidebar"] label { color: #1e3a8a !important; font-weight: 700; }
    div[data-testid="stVerticalBlock"] > div { background-color: white !important; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGICA DI CARICAMENTO E CALCOLO VETTORIALE ---

@st.cache_data
def load_and_process_data():
    try:
        # 1. Carica i vettori atomici (Fonte principale materiali)
        df = pd.read_csv("AF_vectors.csv")
        
        # 2. Carica il Database Elementare
        db = pd.read_csv("Materials Database 1.csv")
        db = db.dropna(subset=['Z'])
        db['Z'] = db['Z'].astype(int)
        
        # Pulizia colonne numeriche elementari
        def clean_num(col):
            return pd.to_numeric(db[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        
        db['Prod_Val'] = clean_num('World production (tons per year)')
        db['Res_Val'] = clean_num('World reserve (tons)')
        db['Risk_Val'] = clean_num('Supply risk')
        db['HHI_Val'] = clean_num('HHI')
        db['ESG_Val'] = clean_num('ESG')

        # Creazione Vettori Proprietà (Z 1-118)
        def get_prop_vec(col):
            return db.set_index('Z')[col].reindex(range(1, 119)).fillna(0).values

        v_prod = get_prop_vec('Prod_Val')
        v_res = get_prop_vec('Res_Val')
        v_risk = get_prop_vec('Risk_Val')
        v_hhi = get_prop_vec('HHI_Val')
        v_esg = get_prop_vec('ESG_Val')

        # Matrice Frazioni Atomiche (N x 118)
        af_cols = [f'AF_{i}' for i in range(1, 119)]
        af_matrix = df[af_cols].fillna(0).values

        # PRODOTTO VETTORIALE (Dot Product)
        df['Calc_Production'] = af_matrix @ v_prod
        df['Calc_Reserves'] = af_matrix @ v_res
        df['Calc_Supply_Risk'] = af_matrix @ v_risk
        df['Calc_HHI'] = af_matrix @ v_hhi
        df['Calc_ESG'] = af_matrix @ v_esg

        # Carica dati S1-S10 se non presenti (Join con file ranking se necessario)
        # Assumiamo che AF_vectors sia già il file pronto
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Errore caricamento file: {e}")
        return None

# --- 3. MOTORE DI CALCOLO RANKING ---

def generate_linear_scores(n):
    return [round((i + 1) / n, 2) for i in range(n)]

def assign_tiered_scores(df, col, n, thresholds):
    scores = generate_linear_scores(n)
    assigned = pd.Series(scores[0], index=df.index, dtype=float)
    for i in range(len(thresholds)):
        assigned[df[col] >= thresholds[i]] = scores[i+1]
    return assigned

# --- 4. INTERFACCIA ---
df = load_and_process_data()

if df is not None:
    st.sidebar.markdown('<p class="sidebar-title">Global Settings</p>', unsafe_allow_html=True)
    manual_t = {'P1': [], 'P2': [], 'P3': []}
    is_valid = True
    
    with st.sidebar:
        # SEZIONE 1: PERFORMANCE
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        # P1
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        sc_t = generate_linear_scores(sf_t)
        for i in range(sf_t - 1):
            val = st.number_input(f"Threshold Score {sc_t[i+1]} (P1)", value=int(350+(i*50)), min_value=350, step=1, format="%d", key=f"p1_{i}")
            manual_t['P1'].append(float(val))
        # P2/P3
        for p_label, p_key, def_idx, def_val in [("Magnetization", "P2", 1, 0.4), ("Coercivity", "P3", 3, 0.4)]:
            st.markdown(f"**{p_label}**")
            sf = st.selectbox(f"Subcategories ({p_key})", [2, 3, 4, 5], index=def_idx, key=f"sf_{p_key}")
            sc = generate_linear_scores(sf)
            for i in range(sf - 1):
                v = st.number_input(f"Threshold Score {sc[i+1]} ({p_key})", value=def_val+(i*0.2), min_value=def_val, key=f"t_{p_key}_{i}")
                manual_t[p_key].append(v)
            if p_key == "P2": sf_m = sf
            else: sf_c = sf

        # Controllo validità soglie
        for k in ['P1', 'P2', 'P3']:
            if any(manual_t[k][i] >= manual_t[k][i+1] for i in range(len(manual_t[k])-1)):
                st.error(f"Error: {k} thresholds must be ascending!")
                is_valid = False

        # SEZIONE 2: PESI
        st.markdown('<div class="blue-section-header"><p>2. Performance Coefficients</p></div>', unsafe_allow_html=True)
        w_p1 = st.slider("Weight P1", 0.0, 1.0, 0.33)
        w_p2 = st.slider("Weight P2", 0.0, 1.0-w_p1, 0.33)
        w_p3 = round(max(0.0, 1.0 - (w_p1 + w_p2)), 2)

        # SEZIONE 3: SCALABILITY SETTINGS (INTERATTIVITÀ RICHIESTA)
        st.markdown('<div class="blue-section-header"><p>3. Scalability Map View</p></div>', unsafe_allow_html=True)
        color_metric = st.selectbox("Map Color Metric:", 
                                    ["Sustainability Score (OSS)", "Supply Risk", "HHI", "ESG"],
                                    index=0)
        metric_map = {
            "Sustainability Score (OSS)": "OSS",
            "Supply Risk": "Calc_Supply_Risk",
            "HHI": "Calc_HHI",
            "ESG": "Calc_ESG"
        }

    # --- CALCOLI CORE ---
    if is_valid:
        # OPS
        p1_s = assign_tiered_scores(df, 'P1', sf_t, manual_t['P1'])
        p2_s = assign_tiered_scores(df, 'P2', sf_m, manual_t['P2'])
        p3_s = assign_tiered_scores(df, 'P3', sf_c, manual_t['P3'])
        df['OPS'] = np.power(p1_s, w_p1) * np.power(p2_s, w_p2) * np.power(p3_s, w_p3)
        
        # OSS (Fisso come media geometrica S1-S10)
        s_cols = [f'S{i}' for i in range(1, 11)]
        if all(c in df.columns for c in s_cols):
            s_data = df[s_cols].apply(pd.to_numeric, errors='coerce').fillna(0.1).to_numpy()
            df['OSS'] = np.exp(np.mean(np.log(np.clip(s_data, 1e-3, 1.0)), axis=1))
        else: df['OSS'] = 0.5

        # --- VISUALIZZAZIONE TABS ---
        t1, t2, t3 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map", "🔬 Stability Analysis"])

        with t1:
            colA, colB = st.columns([2, 1])
            pts = df[['OPS', 'OSS']].to_numpy()
            efficient = np.ones(pts.shape[0], dtype=bool)
            for i, c in enumerate(pts):
                if efficient[i]: efficient[i] = not np.any(np.all(pts >= c, axis=1) & np.any(pts > c, axis=1))
            df['Status'] = np.where(efficient, 'Optimal Choice', 'Standard')
            
            with colA:
                fig = px.scatter(df, x='OPS', y='OSS', color='Status', hover_name='Material_Name',
                                 color_discrete_map={'Optimal Choice': '#1e3a8a', 'Standard': '#cbd5e1'})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with colB:
                st.markdown("**Top Pareto Materials**")
                st.dataframe(df[efficient].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS']], use_container_width=True, height=500)

        with t2:
            st.markdown("### Resource Availability vs. Supply Resilience")
            st.caption(f"Coloring by: {color_metric}")
            
            # Filtro logaritmico per evitare 0
            df_plot = df.copy()
            df_plot['Calc_Production'] = df_plot['Calc_Production'].clip(lower=1e-1)
            df_plot['Calc_Reserves'] = df_plot['Calc_Reserves'].clip(lower=1e-1)
            
            fig_sc = px.scatter(df_plot, x='Calc_Reserves', y='Calc_Production', 
                                color=metric_map[color_metric],
                                size=np.where(efficient, 15, 8),
                                symbol=np.where(efficient, 'star', 'circle'),
                                hover_name='Material_Name',
                                hover_data=['Chemical_Formula', 'Calc_Supply_Risk', 'Calc_HHI'],
                                log_x=True, log_y=True,
                                color_continuous_scale="Viridis",
                                labels={'Calc_Reserves': 'Global Reserves (t)', 'Calc_Production': 'Annual Production (t/yr)'})
            
            fig_sc.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.info("💡 Stars represent materials currently on the Pareto Frontier (Tab 1).")

        with t3:
            opts = df[efficient]['Material_Name'].unique()
            if len(opts) > 0:
                sel = st.selectbox("Select material:", opts)
                if st.button("Run Simulation ⚡"):
                    idx = df[df['Material_Name'] == sel].index[0]
                    rng = np.random.default_rng()
                    W_sim = rng.dirichlet(np.array([w_p1, w_p2, w_p3])*50 + 1, 1000)
                    s_vec = np.array([p1_s[idx], p2_s[idx], p3_s[idx]])
                    c_ops = np.exp(np.dot(W_sim, np.log(s_vec + 1e-9)))
                    fig_mc = px.scatter(x=c_ops, y=[df.loc[idx, 'OSS']]*1000, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                    fig_mc.add_trace(go.Scatter(x=[df.loc[idx, 'OPS']], y=[df.loc[idx, 'OSS']], mode='markers', marker=dict(color='red', size=12, symbol='star')))
                    st.plotly_chart(fig_mc, use_container_width=True)
