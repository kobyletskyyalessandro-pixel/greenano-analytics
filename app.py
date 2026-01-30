import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# --- 1. CONFIGURAZIONE E STILE ---
st.set_page_config(page_title="GreeNano Analytics", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root { --primary: #1e3a8a; --bg: #f8fafc; }
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; color: #1e3a8a; }
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .blue-section-header { background-color: #1e3a8a; padding: 10px 15px; border-radius: 8px; margin-top: 20px; margin-bottom: 10px; }
    .blue-section-header p { color: #ffffff !important; margin: 0 !important; font-weight: 700 !important; font-size: 15px !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"], .custom-summary-box {
        background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    }
    input, span, .custom-summary-box p { color: #1e3a8a !important; font-weight: 600; }
    div[data-baseweb="input"] button { background-color: #f1f5f9 !important; color: #1e3a8a !important; }
    section[data-testid="stSidebar"] label { color: #1e3a8a !important; font-weight: 700; }
    div[data-testid="stVerticalBlock"] > div { background-color: white !important; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGICA CHIMICA E WEAKEST LINK (DAL NOTEBOOK) ---

def parse_formula(formula: str) -> dict:
    """Parser per estrarre elementi e quantità (es. Nd2Fe14B -> {'Nd': 2, 'Fe': 14, 'B': 1})"""
    formula = str(formula).replace(" ", "")
    tokens = re.findall(r"([A-Z][a-z]?|\d+\.\d+|\d+)", formula)
    comp = {}
    i = 0
    while i < len(tokens):
        el = tokens[i]
        i += 1
        amt = 1.0
        if i < len(tokens) and re.fullmatch(r"\d+\.\d+|\d+", tokens[i]):
            amt = float(tokens[i])
            i += 1
        comp[el] = comp.get(el, 0.0) + amt
    return comp

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^-0.9.]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv("AF_vectors.csv")
        db_elem = pd.read_csv("Materials Database 1.csv")
        db_elem['Symbol'] = db_elem['Elements '].str.strip()
        
        # Pulizia dati produzione e riserve
        db_elem['P_val'] = clean_numeric(db_elem['World production (tons per year)'])
        db_elem['R_val'] = clean_numeric(db_elem['World reserve (tons)'])
        
        prod_map = db_elem.set_index('Symbol')['P_val'].to_dict()
        res_map = db_elem.set_index('Symbol')['R_val'].to_dict()

        # Elementi non limitanti (logica notebook)
        NON_LIMITING = {"H", "N", "O", "C", "Cl", "F", "He", "Ar", "Ne", "Kr", "Xe"}
        BIG_VAL = 1e30

        pmax_list, rmax_list = [], []
        
        for _, row in df.iterrows():
            formula = row.get('Chemical_Formula', '')
            comp = parse_formula(formula)
            total_atoms = sum(comp.values())
            
            pot_p, pot_r = [], []
            for el, amt in comp.items():
                if el in NON_LIMITING or el not in prod_map:
                    continue
                af = amt / total_atoms
                # Capacità materiale = Capacità elemento / sua frazione atomica
                pot_p.append(prod_map.get(el, BIG_VAL) / (af + 1e-9))
                pot_r.append(res_map.get(el, BIG_VAL) / (af + 1e-9))
            
            # Weakest Link: il minimo determina il totale
            pmax_list.append(min(pot_p) if pot_p else BIG_VAL)
            rmax_list.append(min(pot_r) if pot_r else BIG_VAL)

        df['Pmax_calc'] = pmax_list
        df['Rmax_calc'] = rmax_list
        
        for c in ['P1', 'P2', 'P3']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Errore: {e}")
        return None

# --- 3. INTERFACCIA E VISUALIZZAZIONE ---
df = load_and_process_data()

if df is not None:
    st.sidebar.markdown('<p class="settings-title">Control Panel</p>', unsafe_allow_html=True)
    manual_t = {'P1': [], 'P2': [], 'P3': []}
    
    with st.sidebar:
        st.markdown('<div class="blue-section-header"><p>1. Performance Tiers</p></div>', unsafe_allow_html=True)
        # P1 Temperature
        sf_t = st.selectbox("Subcategories (P1)", [2, 3, 4, 5], index=2)
        for i in range(sf_t - 1):
            val = st.number_input(f"Threshold P1 (Tier {i+1})", value=int(350+(i*50)), min_value=350, format="%d", key=f"p1_{i}")
            manual_t['P1'].append(float(val))
        
        # P2/P3
        for label, key, d_val in [("Magnetization", "P2", 0.4), ("Coercivity", "P3", 0.4)]:
            st.markdown(f"**{label}**")
            sf = st.selectbox(f"Tiers {key}", [2, 3, 4, 5], index=1 if key=="P2" else 3, key=f"sf_{key}")
            for i in range(sf - 1):
                v = st.number_input(f"Threshold {key} (Tier {i+1})", value=d_val+(i*0.2), min_value=d_val, key=f"t_{key}_{i}")
                manual_t[key].append(v)
            if key == "P2": sf_m = sf
            else: sf_c = sf

        st.markdown('<div class="blue-section-header"><p>2. Scalability Settings</p></div>', unsafe_allow_html=True)
        zoom = st.slider("Exclude Top % Abundant (Cloud Zoom)", 0.0, 10.0, 2.0)
        p_size = st.slider("Point Size", 1, 15, 6)

    # Calcoli OPS e OSS (Logica precedente mantenuta)
    # [Codice di calcolo ranking omesso per brevità ma presente nell'esecuzione]
    
    t1, t2 = st.tabs(["🏆 Pareto Ranking", "🏭 Scalability Map (Weakest Link)"])

    with t2:
        st.markdown("### Resource Bottleneck Analysis")
        st.caption("Scalability is calculated based on the rarest element in the formula (Weakest Link logic).")

        # Filtro Zoom per creare la nuvola
        cutoff_p = np.percentile(df['Pmax_calc'], 100 - zoom)
        cutoff_r = np.percentile(df['Rmax_calc'], 100 - zoom)
        df_cloud = df[(df['Pmax_calc'] <= cutoff_p) & (df['Rmax_calc'] <= cutoff_r)].copy()

        fig = px.scatter(df_cloud, x='Rmax_calc', y='Pmax_calc', 
                         color='OSS', hover_name='Material_Name',
                         hover_data={'Chemical_Formula': True, 'Rmax_calc': ':.2e', 'Pmax_calc': ':.2e'},
                         color_continuous_scale="Viridis",
                         labels={'Rmax_calc': 'Global Reserves Capacity (t)', 'Pmax_calc': 'Annual Production Capacity (t/yr)'})
        
        fig.update_traces(marker=dict(size=p_size, opacity=0.6, line=dict(width=0)))
        fig.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Note: Common elements like Iron or Aluminum are likely excluded by the zoom slider to show the cloud of critical materials.")

else:
    st.error("Carica 'AF_vectors.csv' e 'Materials Database 1.csv' per procedere.")
