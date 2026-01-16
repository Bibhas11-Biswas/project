import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
FILE_PATH = 'augmented_maintenance_data.csv'
TARGET = 'time_to_failure'
SPIKE_THRESHOLD_PCT = 80 
FEATURES = ['vibration', 'temperature', 'pressure', 'stress_index', 'vibration_exceedance_pct']

def backend_analysis():
    print("="*90)
    print("PIPEGUARD AI: NON-LINEAR DIAGNOSTIC SYSTEM")
    print("="*90)

    # 1. DATA LOADING
    try:
        df = pd.read_csv(FILE_PATH)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='T')
    except Exception as e:
        print(f"✗ Load Error: Ensure '{FILE_PATH}' is in the folder.")
        return

    # 2. FEATURE ENGINEERING
    for col in ['vibration', 'temperature', 'pressure']:
        df[f'{col}_roll_mean'] = df[col].rolling(window=5).mean()

    df['vibration_exceedance_pct'] = ((df['vibration'] - df['vibration_roll_mean']) / 
                                      (df['vibration_roll_mean'] + 1e-6)) * 100
    df['stress_index'] = df['pressure'] * (df['vibration'] ** 2)
    df = df.dropna()

    # 3. TRAINING INTELLIGENT ENSEMBLE
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ensemble = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    ensemble.fit(X_train_scaled, y_train)

    # 4. HISTORICAL ANOMALY LOG
    print(f"\n[HISTORICAL ANALYSIS] Scanning for Spikes > {SPIKE_THRESHOLD_PCT}%")
    spikes = df[df['vibration_exceedance_pct'] > SPIKE_THRESHOLD_PCT].tail(8)
    if not spikes.empty:
        spike_preds = ensemble.predict(scaler.transform(spikes[FEATURES]))
        for i, (idx, row) in enumerate(spikes.iterrows()):
            print(f"{str(row['timestamp'])[:19]:<22} | {row['vibration_exceedance_pct']:>9.1f}% | {spike_preds[i]:>9.1f} Days")

    # 5. LIVE MANUAL DIAGNOSTIC (MODIFIED SECTION)
    # --- THIS IS THE SECTION TO REPLACE ---
    print("\n" + "-"*40)
    print("🚀 LIVE MANUAL SENSOR INPUT (RUL CHECK)")
    print("-"*40)
    
    try:
        v_mean = df['vibration'].mean() # Get context from CSV
        
        v = float(input(f"Input Vibration (mm/s) [Avg is {v_mean:.2f}]: "))
        t = float(input("Input Temperature (°C): "))
        p = float(input("Input Pressure (PSI): "))

        s_idx = p * (v ** 2)
        # Calculate context-aware exceedance instead of hardcoding '0'
        m_exceedance = ((v - v_mean) / (v_mean + 1e-6)) * 100
        
        manual_data = pd.DataFrame([[v, t, p, s_idx, m_exceedance]], columns=FEATURES)
        
        # Predict RUL
        rul_prediction = ensemble.predict(scaler.transform(manual_data))[0]

        print("\n" + "="*45)
        print(f"ANALYSIS CONTEXT: Vibration is {m_exceedance:.1f}% from mean.")
        print(f"PREDICTED RUL: {round(rul_prediction, 1)} DAYS")
        
        if rul_prediction < 60:
            print("STATUS: 🚨 CRITICAL - Inspect within 24 hours.")
        elif rul_prediction < 180:
            print("STATUS: ⚠️ WARNING - Plan repair in next cycle.")
        else:
            print("STATUS: ✅ STABLE - Routine monitoring only.")
        print("="*45)
            
    except ValueError:
        print("✗ Input Error: Please enter numbers only.")

    # Save assets for UI
    joblib.dump(ensemble, 'trained_pipe_model.pkl')
    joblib.dump(scaler, 'pipe_scaler.pkl')
    print("\n✓ Model and Scaler exported successfully!")

if __name__ == "__main__":
    backend_analysis()