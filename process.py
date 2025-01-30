import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate synthetic astronomical signal
def generate_signal(samples=1000, freq1=50, freq2=120, noise_level=0.2):
    t = np.linspace(0, 1, samples)
    signal1 = np.sin(2 * np.pi * freq1 * t)
    signal2 = np.sin(2 * np.pi * freq2 * t)
    noise = noise_level * np.random.randn(samples)
    mixed_signal = signal1 + signal2 + noise
    return t, mixed_signal

# Apply FastICA for signal separation
def separate_signal(mixed_signal, components=2):
    ica = FastICA(n_components=components)
    reshaped_signal = mixed_signal.reshape(-1, 1)
    separated_signals = ica.fit_transform(reshaped_signal)
    return separated_signals

# Plot results
def plot_signals(t, mixed_signal, separated_signals):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, mixed_signal, label="Mixed Signal", color='b')
    plt.legend()
    plt.subplot(2, 1, 2)
    for i, sig in enumerate(separated_signals.T):
        plt.plot(t, sig, label=f"Separated Signal {i+1}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    t, mixed_signal = generate_signal()
    separated_signals = separate_signal(mixed_signal)
    plot_signals(t, mixed_signal, separated_signals)
