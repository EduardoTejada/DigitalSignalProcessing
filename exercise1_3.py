import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Função gdel (Exercício 1.1)
def gdel(x, n, Lfft):
    X = np.fft.fft(x, Lfft)
    dXdw = np.fft.fft(n * x, Lfft)
    gd = np.fft.fftshift(np.real(dXdw / X))
    w = (2 * np.pi / Lfft) * np.arange(Lfft) - np.pi
    return gd, w

# Parâmetros
nn = np.arange(-64, 64)  # Vetor de tempo
impulse = (nn == 0).astype(float)  # Impulso em n=0

# Sistema causal Hc(z) = 1 / (1 - 0.77z^-1)
b = [1.0]                # Numerador
a_causal = [1.0, -0.95]  # Denominador (causal)
hc = signal.lfilter(b, a_causal, impulse)  # Resposta ao impulso causal

# Sistema anticausal Ha(z) = Hc(1/z) = 1 / (1 - 0.95z)
a_anticausal = [1.0, -0.95]  # Denominador (anticausal)

# Implementação do filtro anticausal (3 passos):
# 1. Reverte o sinal de entrada: x[-n]
# 2. Filtra causalmente com Hc(z)
# 3. Reverte a saída: y[n] = y_causal[-n]
ha = np.flip(signal.lfilter(b, a_anticausal, np.flip(impulse)))

plt.figure(figsize=(14, 10))

# Subplot 1: Respostas ao impulso
plt.subplot(3, 1, 1)
plt.stem(nn, hc, 'r', markerfmt='ro', label="Causal (Hc(z))", basefmt='k')
plt.stem(nn, ha, 'b', markerfmt='bx', label="Anticausal (Ha(z))", basefmt='k')
plt.xlim(-30, 30); plt.title("Resposta ao Impulso: Causal vs Anticausal")
plt.xlabel("n"); plt.ylabel("Amplitude"); plt.grid(); plt.legend()

# Subplot 2: Magnitude da resposta em frequência
w, Hc = signal.freqz(b, a_causal, worN=1024)
_, Ha = signal.freqz(b, a_anticausal, worN=1024)
plt.subplot(3, 1, 2)
plt.plot(w, 20 * np.log10(np.abs(Hc)), 'r', label="Causal")
plt.plot(w, 20 * np.log10(np.abs(Ha)), 'b--', label="Anticausal")
plt.title("Magnitude da Resposta em Frequência")
plt.xlabel("Frequência [rad/sample]"); plt.ylabel("dB")
plt.grid(); plt.legend()

# Subplot 3: Atraso de grupo
gd_causal, w_gd = gdel(hc, nn, 1024)
gd_anticausal, _ = gdel(ha, nn, 1024)
plt.subplot(3, 1, 3)
plt.plot(w_gd, gd_causal, 'r', label="Causal")
plt.plot(w_gd, gd_anticausal, 'b', label="Anticausal")
plt.title("Atraso de Grupo"); plt.xlabel("Frequência [rad/sample]")
plt.ylabel("Amostras"); plt.grid(); plt.legend()

plt.tight_layout()
plt.show()