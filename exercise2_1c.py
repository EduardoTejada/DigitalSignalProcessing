import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import iir

# Parâmetros do filtro
a = 0.77  # polo causal
a_inv = 1/a  # polo anticausal

# Definir o intervalo de tempo
n = np.arange(-64, 64)  # -64 <= n <= 63
delta = (n == 0).astype(float)  # impulso unitário

# Filtro causal Hc(z) = 1/(1 - 0.77z⁻¹)
h_causal = iir.causal_filter([1], [1, -a], delta)

# Filtro anticausal Ha(z) = 1/(1 - 0.77z)
h_cascade = iir.filtrev([1], [1, -a], h_causal)

"""Calcula e plota a resposta em frequência a partir da resposta ao impulso"""
Lfft = 1024  # Tamanho da FFT

# Calcular a resposta em frequência via FFT
w, H = iir.resposta_em_frequencia_fft(h_cascade)
H_mag = np.abs(H)
H_phase = np.unwrap(np.angle(H))  # Fase desenrolada

# Calcular o atraso de grupo (derivada negativa da fase)
group_delay, w = iir.gdel(h_cascade, n)

# Shift para plotar de -π a π
H_mag_shifted = np.fft.fftshift(H_mag)
group_delay_shifted = np.fft.fftshift(group_delay)
w_shifted = np.linspace(-np.pi, np.pi, Lfft, endpoint=False)

# Plotar
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(w_shifted, H_mag_shifted)
plt.title('Magnitude da Resposta em Frequência (Cascata)')
plt.xlabel('Frequência [rad]')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(w_shifted, group_delay_shifted)
plt.title('Atraso de Grupo (Cascata)')
plt.xlabel('Frequência [rad]')
plt.ylabel('Amostras')
plt.grid(True)

plt.tight_layout()
plt.show()