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
    
    /* 1. RESET GENERALE */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        color: #1e3a8a;
    }
    html, body, .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: #f8fafc; 
        color: #1e3a8a; 
    }
    
    /* 2. SIDEBAR BIANCA */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* 3. INPUT BOX & SELECTBOX: STILE DEFAULT PULITO (Bianco/Grigio) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important; /* Bordo Grigio Chiaro */
        border-radius: 8px !important;
        color: #1e3a8a !important;
    }
    
    /* Testo dentro gli input */
    input[type="number"], div[data-baseweb="select"] span {
        color: #1e3a8a !important; /* Blu scuro */
        -webkit-text-fill-color: #1e3a8a !important;
        caret-color: #1e3a8a !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    
    /* Icone grigie */
    div[data-baseweb="select"] svg, 
    div[data-baseweb="input"] button {
        color: #64748b !important;
        fill: #64748b !important;
        background-color: transparent !important;
    }

    /* 4. LABEL SIDEBAR */
    section[data-testid="stSidebar"] label {
        color: #1e3a8a !important;
        font-weight: 700;
        font-size: 14px;
    }
    
    /* 5. TITOLI & CARD */
    h1, h2, h3, h4 { color: #1e3a8a !important; font-weight: 800; }
    
    div[data-testid="stVerticalBlock"] > div { 
        background-color: white !important; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1e3a8a !important;
    }
    
    /* 6. BOTTONI AZIONE */
    div.stButton > button {
        background-color: #1e3a8a !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #2563eb !important;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER HEADER (TITOLI BLU) ---
def blue_header(text):
    st.markdown(f"""
    <div style="
        background-color: #1e3a8a; 
        color: white; 
        padding: 10px 15px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        {text}
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
<div style="padding: 15px; border-left: 5px solid #1e3a8a; background-color: white; margin-bottom: 25px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <h4 style="margin:0; color:#1e3a8a;">🚀 Calculation Engine</h4>
    <p style="margin:0; color:#475569;">Configure <b>Tiers (Subcategories)</b> and <b>Weights</b> to rank materials dynamically.</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

if df is not None:
    # --- SIDEBAR (NESSUN TITOLO "SETTINGS") ---
    
    # 1. TIERS
    blue_header("1. Performance Tiers")
    with st.sidebar:
        sf_t = st.selectbox("Tiers for P1 (Temp)", [2, 3, 4, 5], index=2) 
        sf_m = st.selectbox("Tiers for P2 (Mag)", [2, 3, 4, 5], index=1)
        sf_c = st.selectbox("Tiers for P3 (Coerc)", [2, 3, 4, 5], index=3)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. WEIGHTS
    blue_header("2. Coefficients")
    with st.sidebar:
        w_p1 = st.slider("Weight P1 (Temp)", 0.0, 1.0, 0.33)
        rem = 1.0 - w_p1
        w_p2 = st.slider("Weight P2 (Mag)", 0.0, max(0.0, rem), min(0.33, rem))
        w_p3 = max(0.0, 1.0 - (w_p1 + w_p2))
        
        # Riassunto Pesi (Box Blu Scuro)
        st.markdown(f"""
        <div style="background-color: #1e3a8a; color: white; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px; font-weight: 500;">
            Temp: {w_p1:.2f} | Mag: {w_p2:.2f} | Coerc: {w_p3:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.info("Sustainability data is fixed (LCA).")

    # --- CALCOLO ---
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
        tab1, tab2, tab3 = st.tabs(["🏆 Pareto Ranking", "🏭 Supply Chain", "🔬 Stability"])

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
                fig.update_layout(template="plotly_white", xaxis_title="OPS (Tiered Score)", yaxis_title="OSS (Sustainability)")
                st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                st.markdown("**Top Materials**")
                st.dataframe(df[mask].sort_values(by="OPS", ascending=False)[['Material_Name', 'OPS', 'OSS', 'P1']], use_container_width=True, height=500)

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
                    
                    ps1 = assign_tiered_scores(df, 'P1', sf_t).loc[idx]
                    ps2 = assign_tiered_scores(df, 'P2', sf_m).loc[idx]
                    ps3 = assign_tiered_scores(df, 'P3', sf_c).loc[idx]
                    
                    c_ops = np.exp(np.dot(W_ops, np.log([ps1, ps2, ps3] + np.array([1e-9]*3))))
                    W_oss = rng.dirichlet(np.ones(10)*20, 1000)
                    s_oss = df.loc[idx, s_cols].to_numpy(dtype=float)
                    c_oss = np.exp(np.dot(W_oss, np.log(s_oss+1e-9)))
                    
                    fig_mc = px.scatter(x=c_ops, y=c_oss, opacity=0.3, color_discrete_sequence=['#1e3a8a'])
                    st.plotly_chart(fig_mc, use_container_width=True)

    else:
        st.error("CSV Missing P1/P2/P3 columns")
else:
    st.warning("Upload CSV")
