import numpy as np
import pandas as pd

def generate_synthetic_rul_data(
    n_units=200,
    min_life=30,
    max_life=180,
    noise_level=0.12
):
    records = []
    
    for unit_id in range(n_units):
        total_life = np.random.randint(min_life, max_life + 1)
        
        for t in range(total_life + 1):
            rul = total_life - t
            
            # Degradation physics: sensors drift as failure approaches
            vib_base = 1.0 + 9.0 * (1 - rul / total_life) ** 2      # accelerates near end
            temp_base = 50 + 45 * (1 - rul / total_life)
            press_base = 40 - 25 * (1 - rul / total_life)           # drops due to leaks/blockage
            
            # Add realistic noise
            vibration = np.clip(vib_base + np.random.normal(0, noise_level * 4), 0.1, 15)
            temperature = np.clip(temp_base + np.random.normal(0, noise_level * 8), 50, 100)
            pressure = np.clip(press_base + np.random.normal(0, noise_level * 6), 5, 45)
            
            records.append({
                'vibration': vibration,
                'temperature': temperature,
                'pressure': pressure,
                'time_to_failure': rul
            })
    
    return pd.DataFrame(records)

# Generate and save
df = generate_synthetic_rul_data(n_units=150)
df.to_excel('pipe_data.xlsx', index=False)
print(f"✅ Generated {len(df)} rows of realistic RUL data.")
print("📁 Saved as 'pipe_data.xlsx'")