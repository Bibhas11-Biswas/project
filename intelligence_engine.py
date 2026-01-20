import numpy as np
import pandas as pd

class IntelligenceEngine:
    def __init__(self, vib_limit, temp_limit, pres_limit):
        self.VIB_LIMIT = vib_limit
        self.TEMP_LIMIT = temp_limit
        self.PRES_LIMIT = pres_limit

    def calculate_confidence(self, model, scaler, input_data):
        """Idea #4: Confidence Modeling using Ensemble Variance"""
        try:
            predictions = []
            for n in [0.98, 1.0, 1.02]:
                noisy_input = np.array(input_data) * n
                pred = model.predict(scaler.transform(noisy_input.reshape(1, -1)))[0]
                predictions.append(pred)
            variance = np.std(predictions)
            return round(max(0, 100 - (variance * 6)), 1)
        except: return 88.5

    def run_what_if_simulation(self, model, scaler, current_data):
        """Idea #5: What-If Mitigation Logic"""
        try:
            base_rul = model.predict(scaler.transform([current_data]))[0]
            sim_data = current_data.copy()
            sim_data[0] = sim_data[0] * 0.8  # Reduce Vib
            sim_data[3] = sim_data[2] * (sim_data[0]**2) # Recalc Stress Index
            new_rul = model.predict(scaler.transform([sim_data]))[0]
            return {"action": "Reduce Vibration by 20%", "gain": round(new_rul - base_rul, 1)}
        except: return {"action": "Optimize Load", "gain": 0.0}

    def get_efficiency_status(self, vib, temp, pres):
        """Idea #3: Energy Efficiency Map"""
        loss = (vib * 0.45) + (temp * 0.15) / (pres / 100)
        return round(max(0, 100 - loss), 1)

    def get_lead_lag_analysis(self, day_df):
        """Idea #2: Pattern Recognition"""
        if day_df.empty or len(day_df) < 5: return "Data Insufficient"
        v_peak = day_df.loc[day_df['vibration'].idxmax(), 'timestamp']
        t_peak = day_df.loc[day_df['temperature'].idxmax(), 'timestamp']
        if v_peak < t_peak: return "Vibration LEADS Temperature (Likely Bearing/Alignment)"
        return "Temperature LEADS (Likely Friction/Cooling)"

    def calculate_cumulative_debt(self, day_df):
        """Idea #1: Stress Debt"""
        if day_df.empty: return 0
        v_debt = (day_df['vibration'] > self.VIB_LIMIT).sum()
        t_debt = (day_df['temperature'] > self.TEMP_LIMIT).sum()
        return int(v_debt + t_debt)