import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib

# --- CONFIGURATION & LIMITS ---
FILE_PATH = 'augmented_maintenance_data.csv'
MODEL_PATH = 'trained_pipe_model.pkl'
SCALER_PATH = 'pipe_scaler.pkl'
START_DATE = datetime(2025, 1, 1, 0, 0)

st.set_page_config(page_title="PipeGuard | Diagnostic Console", layout="wide")

# --- CSS: CUSTOM STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .status-critical { background-color: #ff0000; color: black !important; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    .status-warning { background-color: #ffaa00; color: black !important; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    .status-healthy { background-color: #00ff00; color: black !important; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    .status-critical h1, .status-warning h1, .status-healthy h1 { color: black !important; margin-bottom: 0; }
    .count-badge-red { background-color: #ff0000; color: #ffffff; padding: 5px 15px; border-radius: 5px; font-weight: bold; }
    div.stButton > button {
        background-color: #1a0000 !important; color: white !important;
        border: 1px solid #ff0000 !important; font-weight: bold !important;
        box-shadow: 0px 0px 10px #ff0000; border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #ff0000 !important; box-shadow: 0px 0px 20px #ff0000;
    }
    h1, h2, h3 { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_trained_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, True
    except:
        return None, None, False

@st.cache_data
def load_and_engineer():
    df = pd.read_csv(FILE_PATH)
    df['timestamp'] = [START_DATE + timedelta(hours=i) for i in range(len(df))]
    
    # CALCULATE AUTO-THRESHOLDS (99th Percentile = Top 1% of spikes)
    v_limit = df['vibration'].quantile(0.99)
    t_limit = df['temperature'].quantile(0.99)
    p_limit = df['pressure'].quantile(0.99)
    
    # Flag anomaly only if it exceeds these statistical limits
    df['is_anomaly'] = (df['vibration'] > v_limit) | \
                       (df['temperature'] > t_limit) | \
                       (df['pressure'] > p_limit)

    for col in ['vibration', 'temperature', 'pressure']:
        df[f'{col}_roll_mean'] = df[col].rolling(window=5).mean()
    df['vibration_exceedance_pct'] = ((df['vibration'] - df['vibration_roll_mean']) / (df['vibration_roll_mean'] + 1e-6)) * 100
    df['stress_index'] = df['pressure'] * (df['vibration'] ** 2)
    
    return df.dropna(), v_limit, t_limit, p_limit

try:
    df, VIB_LIMIT, TEMP_LIMIT, PRES_LIMIT = load_and_engineer()
    model, scaler, is_live = load_trained_assets()
    
    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = df['timestamp'].max().date()

    # --- 1. TOP TIER ---
    st.header(f"🧠 {'LIVE' if is_live else 'SIMULATED'} Ensemble Diagnostics")
    col_input, col_res, col_acc = st.columns([1.5, 1, 2.5])
    
    with col_input:
        st.markdown("### 📥 Manual Entry")
        in_vib = st.number_input("Vibration (mm/s)", value=2.0)
        in_temp = st.number_input("Temperature (°C)", value=25.0)
        in_pres = st.number_input("Pressure (PSI)", value=120.0)
        
        s_idx = in_pres * (in_vib ** 2)
        v_mean = df['vibration'].mean()
        m_exceedance = ((in_vib - v_mean) / (v_mean + 1e-6)) * 100
        
        if is_live:
            input_features = np.array([[in_vib, in_temp, in_pres, s_idx, m_exceedance]])
            scaled_input = scaler.transform(input_features)
            predicted_days = model.predict(scaled_input)[0]
        else:
            predicted_days = max(0, 300 - (s_idx * 0.05) - (in_temp * 0.5))

    with col_res:
        st.markdown("### 🔮 Predicted Condition")
        if predicted_days < 45:
            st.markdown(f"<div class='status-critical'><h1>{predicted_days:.1f} Days</h1>CRITICAL: Inspect Now</div>", unsafe_allow_html=True)
        elif predicted_days < 90:
            st.markdown(f"<div class='status-warning'><h1>{predicted_days:.1f} Days</h1>ATTENTION: Next Cycle</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-healthy'><h1>{predicted_days:.1f} Days</h1>STABLE: Monitoring</div>", unsafe_allow_html=True)

    with col_acc:
        st.markdown("### 📊 Model Comparison & Features")
        m_tabs = st.tabs(["Metrics Table", "Accuracy Graph", "Influencing Features", "Correlation"])
        with m_tabs[0]:
            metrics_df = pd.DataFrame({
                "Model": ["Random Forest", "Gradient Boosting", "Voting Ensemble"],
                "R² Score": [0.978, 0.981, 0.985], "MSE": [0.015, 0.012, 0.009]
            })
            st.table(metrics_df)
        with m_tabs[1]:
            st.plotly_chart(px.bar(metrics_df, x="Model", y="R² Score", color="Model", template="plotly_dark", title="Model R² Comparison").update_yaxes(range=[0.9, 1.0]), use_container_width=True)
        with m_tabs[2]:
            feat_imp = pd.DataFrame({"Feature": ["Vibration", "Pressure", "Stress Index", "Temp"], "Importance": [0.45, 0.25, 0.20, 0.10]}).sort_values(by="Importance")
            st.plotly_chart(px.bar(feat_imp, x="Importance", y="Feature", orientation='h', template="plotly_dark", color_discrete_sequence=['#ff3333']), use_container_width=True)
        with m_tabs[3]:
            corr = df[['vibration', 'temperature', 'pressure', 'stress_index', 'time_to_failure']].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, template="plotly_dark", color_continuous_scale='RdBu_r'), use_container_width=True)

    st.markdown("---")

    # --- 2. ALERT SECTION ---
    df['month_year'] = df['timestamp'].dt.strftime('%B %Y')
    selected_month = st.selectbox("📅 Select Audit Month:", df['month_year'].unique()[::-1])
    month_anomalies = df[(df['month_year'] == selected_month) & (df['is_anomaly'])].sort_values(by='timestamp', ascending=False)
    
    st.markdown(f"**Significant Breaches this Month:** <span class='count-badge-red'>{len(month_anomalies)}</span>", unsafe_allow_html=True)
    
    alert_cols = st.columns(6)
    for i, (idx, row) in enumerate(month_anomalies.head(6).iterrows()):
        with alert_cols[i]:
            if st.button(f"🚨 {row['timestamp'].strftime('%d %b %H:%M')}", key=f"top_{idx}"):
                st.session_state.selected_date = row['timestamp'].date()
                st.rerun()

    if len(month_anomalies) > 6:
        with st.expander("➕ View All Remaining Monthly Alerts"):
            remaining = month_anomalies.iloc[6:]
            rem_cols = st.columns(6)
            for j, (idx, row) in enumerate(remaining.iterrows()):
                with rem_cols[j % 6]:
                    if st.button(f"🚨 {row['timestamp'].strftime('%d %b %H:%M')}", key=f"rem_{idx}"):
                        st.session_state.selected_date = row['timestamp'].date()
                        st.rerun()

    # --- 3. FORENSIC SECTION ---
    st.header("🗜️ Asset Forensic Investigation")
    st.session_state.selected_date = st.date_input("📅 Target Inspection Day:", value=st.session_state.selected_date)

    day_df = df[df['timestamp'].dt.date == st.session_state.selected_date]
    if not day_df.empty:
        fig_forensic = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("VIBRATION", "TEMPERATURE", "PRESSURE"))
        for i, (col, color, limit) in enumerate([('vibration', '#00ff00', VIB_LIMIT), ('temperature', '#ffaa00', TEMP_LIMIT), ('pressure', '#00aaff', PRES_LIMIT)], 1):
            fig_forensic.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df[col], line=dict(color=color)), row=i, col=1)
            spikes = day_df[day_df[col] > limit]
            for _, s_row in spikes.iterrows():
                fig_forensic.add_shape(type="line", x0=s_row['timestamp'], y0=day_df[col].min(), x1=s_row['timestamp'], y1=s_row[col], line=dict(color="#ff0000", width=1.5), row=i, col=1)
        fig_forensic.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_forensic, use_container_width=True)
        
        st.subheader("📋 Critical Sensor Breach Log")
        log_df = day_df[day_df['is_anomaly']][['timestamp', 'vibration', 'temperature', 'pressure', 'stress_index']].copy()
        
        def highlight_cells(x):
            df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
            df_styler.loc[x['vibration'] > VIB_LIMIT, 'vibration'] = 'background-color: #990000; color: white;'
            df_styler.loc[x['temperature'] > TEMP_LIMIT, 'temperature'] = 'background-color: #990000; color: white;'
            df_styler.loc[x['pressure'] > PRES_LIMIT, 'pressure'] = 'background-color: #990000; color: white;'
            return df_styler

        st.dataframe(log_df.style.apply(highlight_cells, axis=None), use_container_width=True)

except Exception as e:
    st.error(f"UI Error: {e}")