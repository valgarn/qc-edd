import numpy as np
import matplotlib.pyplot as plt

def simulate_sensor_data(label='healthy', duration=10, sampling_rate=50):
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Healthy subject: smooth, rhythmic movement
    if label == 'healthy':
        signal = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(len(t))
    
    # Early Parkinson's: irregularities, tremor, reduced amplitude
    elif label == 'early_pd':
        tremor = 0.3 * np.sin(2 * np.pi * 5 * t + np.random.uniform(-0.1, 0.1, len(t)))
        bradykinesia = np.sin(2 * np.pi * 1.5 * t) * (0.5 + 0.3 * np.sin(0.2 * np.pi * t))
        noise = 0.1 * np.random.randn(len(t))
        signal = bradykinesia + tremor + noise
    else:
        raise ValueError("Label must be 'healthy' or 'early_pd'")
    
    return t, signal

# Example plot
t, healthy = simulate_sensor_data('healthy')
_, early_pd = simulate_sensor_data('early_pd')

plt.figure(figsize=(10, 4))
plt.plot(t, healthy, label='Healthy')
plt.plot(t, early_pd, label='Early PD', alpha=0.75)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Sensor Value")
plt.title("Simulated Biosensor Time-Series")
plt.savefig("images/sensor_data.png")
