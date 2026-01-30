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
        border: 1px solid #cbd5e1 !important;
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

    /* LABELS SIDEBAR */
    section[data-testid="stSidebar"] label {
        color: #1e3a8a !important;
        font-weight: 700;
        font-size: 13px;
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
    
    .sidebar-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER HEADER (TITOLI BLU) ---
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

# --- MOTORE DI CALCOLO (TIER + MANUAL THRESHOLDS) ---

SF_SCORE_MAP = {
    2: [1.0, 0.5],
    3: [1.0, 0.6, 0.3],
    4: [1.0, 0.75, 0.5, 0.25],
    5: [1.0, 0.8, 0.6, 0.4, 0.2]
}

def assign_manual_tiered_scores(df, col_name, sf_value, manual_thresholds):
    """
    Assegna score basati su soglie manuali definite dall'utente.
    manual_thresholds: Lista di valori (es. [400, 300] per 3 tiers).
                       Val >= 400 -> Score 1
                       Val >= 300 -> Score 2
                       Else -> Score 3
    """
    scores_list = SF_SCORE_MAP.get(sf_value, SF_SCORE_MAP[3])
    
    # Inizializza con lo score più basso
    assigned_scores = pd.Series(scores_list[-1], index=df.index, dtype=float)
    
    # Assegna gli score partendo dal livello più alto
    # manual_thresholds ha N-1 valori. 
    # Esempio 3 Tiers -> 2 Soglie (T1, T2).
    # Se Val >= T1 -> Score[0]
    # Se T2 <= Val < T1 -> Score[1]
    
    # Iteriamo sui livelli alti
    for i, threshold in enumerate(manual_thresholds):
        mask = df[col_name] >= threshold
        # Assegna lo score solo se non è stato già assegnato uno score più alto (logica a cascata implicita dall'ordine)
        # Ma qui usiamo un approccio diretto:
        # Per il primo threshold (Top Tier), assegniamo a tutti quelli sopra.
        # Per i successivi, assegniamo solo a quelli sopra CHE NON SONO già stati assegnati (opzionale se ordiniamo bene)
        
        # Semplice: Assegna lo score a tutti quelli >= threshold. 
        # Poiché iteriamo dal Tier 1 (più alto) a scendere, dobbiamo fare attenzione a non sovrascrivere?
        # No, se iteriamo dal basso verso l'alto (soglie più basse) sovrascriviamo con score migliori.
        pass

    # REFINE LOGIC:
    # 1. Setta tutto al minimo (Score N)
    # 2. Per ogni soglia (dalla più bassa alla più alta), assegna lo score corrispondente a chi la supera.
    
    # Ordiniamo soglie e score per sicurezza (Soglia Bassa -> Score Basso? No.)
    # Soglia Alta -> Score Alto (1.0)
    
    # Esempio: Tiers=3 (Score: 1.0, 0.6, 0.3). Soglie Input: [400, 300] (T1=400, T2=300).
    # Chi è >= 300 prende 0.6.
    # Chi è >= 400 prende 1.0 (sovrascrive 0.6).
    
    # Ordiniamo le soglie e gli score accoppiati
    # Ci servono sf_value-1 soglie.
    # Scores corrispondenti: scores_list[0] (per >= T1), scores_list[1] (per >= T2), etc.
    
    # Applichiamo dal basso (soglia più bassa) all'alto
    # Reverse order di applicazione
    
    thresholds_sorted = sorted(manual_thresholds) # Es. [300, 400]
    # Gli score corrispondenti sono:
    # Se supero 300 (soglia bassa) -> Prendo Score[1] (0.6)
    # Se supero 400 (soglia alta) -> Prendo Score[0] (1.0)
    
    # Quindi mappiamo thresholds_sorted[k] -> scores_list[sf_value - 2 - k]
    # Es. 3 Tiers -> 2 soglie. Sorted: [LowT, HighT].
    # LowT -> Score Medio (0.6) -> Index 1
    # HighT -> Score Alto (1.0) -> Index 0
    
    num_thresholds = len(thresholds_sorted)
    
    for i in range(num_thresholds):
        thresh = thresholds_sorted[i]
        # Score index corresponding: (N_tiers - 2) - i
        # Es. N=3. i=0 (LowT) -> Index 1 (0.6). i=1 (HighT) -> Index 0 (1.0).
        score_idx = (sf_value - 2) - i
        if score_idx >= 0:
            target_score = scores_list[score_idx]
            assigned_scores[df[col_name] >= thresh] = target_score
            
    return assigned_scores

def calculate_ops_manual(df, thresholds_map, tiers_config, weights):
    # thresholds_map: {'P1': [t1, t2...], 'P2': [...]}
    ps1 = assign_manual_tiered_scores(df, 'P1', tiers_config['P1'], thresholds_map['P1'])
    ps2 = assign_manual_tiered_scores(df, 'P2', tiers_config['P2'], thresholds_map['P2'])
    ps3 = assign_manual_tiered_scores(df, 'P3', tiers_config['P3'], thresholds_map['P3'])
    
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
<div style="padding: 15px; border-left: 5px solid #1e3a8a; background-color: white; margin-bottom: 25px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <h4 style="margin:0; color:#1e3a8a;">🚀 Calculation Engine</h4>
    <p style="margin:0; color:#475569;">Define <b>Tiers</b> and their <b>Thresholds</b> to score materials.</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    st.sidebar.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
    
    # Dizionario per salvare le soglie manuali
    manual_thresholds = {'P1': [], 'P2': [], 'P3': []}
    
    # 1. TIERS & THRESHOLDS CONFIG
    with st.sidebar:
        blue_header("1. Performance Configuration")
        
        # --- P1 TEMP ---
        st.markdown("**P1: Temperature (K)**")
        sf_t = st.selectbox("Tiers (P1)", [2, 3, 4, 5], index=2, key='sf_t')
        
        # Genera input dinamici per le soglie
        # Usiamo i quantili come valori di default suggeriti
        p1_vals = df['P1'].sort_values(ascending=False).values
        n_p1 = len(p1_vals)
        for i in range(sf_t - 1):
            # Calcola suggestion
            idx = int(n_p1 * (i + 1) / sf_t)
            default_val = float(p1_vals[idx]) if idx < n_p1 else 0.0
            
            label_text = f"Tier {i+1} Limit (Score {SF_SCORE_MAP[sf_t][i]})"
            val = st.number_input(label_text, value=default_val, key=f"t_p1_{i}")
            manual_thresholds['P1'].append(val)
            
        st.markdown("---")

        # --- P2 MAG ---
        st.markdown("**P2: Magnetization (T)**")
        sf_m = st.selectbox("Tiers (P2)", [2, 3, 4, 5], index=1, key='sf_m')
        p2_vals = df['P2'].sort_values(ascending=False).values
        n_p2 = len(p2_vals)
        for i in range(sf_m - 1):
            idx = int(n_p2 * (i + 1) / sf_m)
            default_val = float(p2_vals[idx]) if idx < n_p2 else 0.0
            val = st.number_input(f"Tier {i+1} Limit (Score {SF_SCORE_MAP[sf_m][i]})", value=default_val, key=f"t_p2_{i}")
            manual_thresholds['P2'].append(val)

        st.markdown("---")

        # --- P3 COERC ---
        st.markdown("**P3: Coercivity (T)**")
        sf_c = st.selectbox("Tiers (P3)", [2, 3, 4, 5], index=3, key='sf_c')
        p3_vals = df['P3'].sort_values(ascending=False).values
        n_p3 = len(p3_vals)
        for i in range(sf_c - 1):
            idx = int(n_p3 * (i + 1) / sf_c)
            default_val = float(p3_vals[idx]) if idx < n_p3 else 0.0
            val = st.number_input(f"Tier {i+1} Limit (Score {SF_SCORE_MAP[sf_c][i]})", value=default_val, key=f"t_p3_{i}")
            manual_thresholds['P3'].append(val)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. WEIGHTS
    with st.sidebar:
        blue_header("2. Coefficients")
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        st.markdown(f"""
        <div style="background-color: white; color: #1e3a8a; padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; text-align: center; margin-top: 10px; font-weight: 600; font-size: 14px;">
            Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.caption("Sustainability data is fixed (LCA).")

    # --- CALCOLO ---
    if all(c in df.columns for c in ['P1', 'P2', 'P3']):
        
        tiers_config = {'P1': sf_t, 'P2': sf_m, 'P3': sf_c}
        weights_perf = [w_p1, w_p2, w_p3]
        
        # CALCOLO OPS CON SOGLIE MANUALI
        df['OPS'] = calculate_ops_manual(df, manual_thresholds, tiers_config, weights_perf)
        
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
                fig.update_layout(template="plotly_white", xaxis_title="OPS (Weighted Performance)", yaxis_title="OSS (Sustainability)")
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Materials**")
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
                    
                    # Recupera score correnti
                    # Per simulazione Monte Carlo, semplifichiamo usando l'OPS corrente come base
                    # Idealmente dovremmo ricalcolare gli score P1, P2, P3 basati sulle soglie manuali per ogni iterazione
                    # Ma le soglie sono fisse, cambiano solo i pesi (W_ops)
                    
                    # 1. Calcola vettori P1, P2, P3 score per l'elemento selezionato
                    p1_score = assign_manual_tiered_scores(df.iloc[[idx]], 'P1', sf_t, manual_thresholds['P1']).values[0]
                    p2_score = assign_manual_tiered_scores(df.iloc[[idx]], 'P2', sf_m, manual_thresholds['P2']).values[0]
                    p3_score = assign_manual_tiered_scores(df.iloc[[idx]], 'P3', sf_c, manual_thresholds['P3']).values[0]
                    
                    s_vec = np.array([p1_score, p2_score, p3_score])
                    
                    # 2. Applica Monte Carlo sui pesi
                    c_ops = np.exp(np.dot(W_ops, np.log(s_vec + 1e-9)))
                    
                    W_oss = rng.dirichlet(np.ones(10)*20, 1000)
                    s_oss = df.loc[idx, s_cols].to_numpy(dtype=float)
                    c_oss = np.exp(np.dot(W_oss, np.log(s_oss+1e-9)))
                    
                    fig_mc = px.scatter(x=c_ops, y=c_oss, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                    st.plotly_chart(fig_mc, use_container_width=True)

    else:
        st.error("CSV Missing P1/P2/P3 columns")
else:
    st.warning("Upload CSV")
