import numpy as np
import matplotlib.pyplot as plt
import os

def generate_mock_results():
    print("Generating out-of-the-box visualization...")
    
    # 1. Time sequence (e.g., 60 seconds)
    time = np.arange(0, 60, 1)
    
    # 2. Simulate Heart Rate (BPM)
    # Normal is 60-100. Let's create a scenario where it spikes.
    base_hr = 75
    hr = base_hr + np.sin(time/5) * 5
    hr[40:] += np.linspace(0, 40, 20) + np.random.normal(0, 2, 20) # Spike at end
    
    # 3. Simulate Respiration Rate (RPM)
    # Normal is 12-20.
    base_rr = 15
    rr = base_rr + np.cos(time/3) * 2
    rr[45:] += np.linspace(0, 15, 15) + np.random.normal(0, 1, 15)
    
    # 4. Simulate the Model's output: Risk Score
    # Risk increases as HR and RR spike.
    risk_score = 0.1 * np.ones_like(time)
    for i in range(len(time)):
        if time[i] > 35:
            risk_score[i] = min(0.95, risk_score[i-1] + 0.05 + np.random.normal(0, 0.02))
        else:
            risk_score[i] = max(0.05, min(0.2, risk_score[i-1] + np.random.normal(0, 0.01)))
            
    # 5. Plotting
    plt.style.use('dark_background')
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Heart Rate
    axs[0].plot(time, hr, color='#ff4d4d', linewidth=2, label='Heart Rate (BPM)')
    axs[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    axs[0].set_ylabel('BPM')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, alpha=0.2)
    axs[0].set_title('Multimodal Physiological Monitoring Output', fontsize=14, pad=20)
    
    # Respiration Rate
    axs[1].plot(time, rr, color='#4da6ff', linewidth=2, label='Resp Rate (RPM)')
    axs[1].axhline(y=20, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_ylabel('RPM')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, alpha=0.2)
    
    # Risk Score
    axs[2].plot(time, risk_score, color='#ffcc00', linewidth=2, label='Clinical Risk Score (0-1)')
    axs[2].fill_between(time, risk_score, 0, color='#ffcc00', alpha=0.3)
    axs[2].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
    axs[2].set_ylabel('Risk Probability')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].legend(loc='upper left')
    axs[2].grid(True, alpha=0.2)
    axs[2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_of_box_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    generate_mock_results()
