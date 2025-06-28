import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parâmetros
nn = np.arange(-64, 64)
b = [1.0]         # Numerador (1)
#a = [1.0, -0.77]  # Denominador (1 - 0.77z^-1)
a = [1.0, -0.95]

# 1. Resposta ao impulso
impulse = (nn == 0).astype(float)
hc = signal.lfilter(b, a, impulse)

# 2. Resposta em frequência (EXATA: usando freqz)
w, H_exato = signal.freqz(b, a, worN=1024, whole=False)

# 3. Resposta em frequência (APROXIMADA: usando FFT da hc[n])
H_approx = np.fft.fft(hc, 1024)[:512]  # Usamos metade por simetria
w_approx = np.linspace(0, np.pi, 512)

# Plot
plt.figure(figsize=(12, 8))

# Subplot 1: Resposta ao impulso
plt.subplot(3, 1, 1)
plt.stem(nn, hc, basefmt='k', linefmt='C0-', markerfmt='C0o')
plt.xlim(-5, 20); plt.title("Resposta ao Impulso (Causal)")
plt.xlabel("n"); plt.ylabel("h_c[n]"); plt.grid()

# Subplot 2: Magnitude da resposta em frequência
plt.subplot(3, 1, 2)
plt.plot(w, 20 * np.log10(np.abs(H_exato)), 'r', label="Exato (freqz)")
plt.plot(w_approx, 20 * np.log10(np.abs(H_approx)), 'b--', label="Aproximado (FFT)")
plt.title("Magnitude da Resposta em Frequência (dB)")
plt.xlabel("Frequência [rad/sample]"); plt.ylabel("dB")
plt.legend(); plt.grid()

# Subplot 3: Atraso de grupo (usando gdel)
def gdel(x, n, Lfft):
    X = np.fft.fft(x, Lfft)
    dXdw = np.fft.fft(n * x, Lfft)
    gd = np.fft.fftshift(np.real(dXdw / X))
    w = (2 * np.pi / Lfft) * np.arange(Lfft) - np.pi
    return gd, w

gd, w_gd = gdel(hc, nn, 1024)
plt.subplot(3, 1, 3)
plt.plot(w_gd, gd); plt.title("Atraso de Grupo")
plt.xlabel("Frequência [rad/sample]"); plt.ylabel("Amostras")
plt.grid()

plt.tight_layout()
plt.show()