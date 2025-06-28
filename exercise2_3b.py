import numpy as np
import matplotlib.pyplot as plt
import iir

# Coeficientes das partes causal e anticausal
b1 = [0.1149, 0.0596, -0.0416]  # numerador causal
a1 = [1, -1.2062, 0.5406]       # denominador causal

# Definir o intervalo de tempo
n = np.arange(-64, 64)  # -64 <= n <= 63
delta = (n == 0).astype(float)  # impulso unitário

# 1. Processar parte causal
y1 = iir.causal_filter(b1, a1, delta)

# 2. Processar parte anticausal
y2 = iir.filtrev(b1, a1, delta)

# 3. Soma das partes
h_parallel = y1 + y2

# Plotar os resultados
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(n, y1, 'b', markerfmt='bo', basefmt=" ")
plt.title('Resposta ao Impulso da Parte Causal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n, h_parallel, 'r', markerfmt='ro', basefmt=" ")
plt.title('Resposta ao Impulso Total (Paralela)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

# Calcular e plotar a resposta em frequência
Lfft = 1024
H = np.fft.fft(h_parallel, Lfft)
H_mag = np.abs(H)
w = np.linspace(0, 2*np.pi, Lfft, endpoint=False)

plt.subplot(3, 1, 3)
plt.plot(w, H_mag)
plt.title('Magnitude da Resposta em Frequência (Paralela)')
plt.xlabel('Frequência [rad]')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()