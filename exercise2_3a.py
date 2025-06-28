import numpy as np
import matplotlib.pyplot as plt
import iir

# Coeficientes do filtro causal Hc(z)
b_causal = [0.1432, 0.0117, 0.1432]  # numerador
a_causal = [1, -1.2062, 0.5406]      # denominador

# Definir o intervalo de tempo
n = np.arange(-64, 64)  # -64 <= n <= 63
delta = (n == 0).astype(float)  # impulso unitário

# 1. Aplicar o filtro causal
h_causal = iir.causal_filter(b_causal, a_causal, delta)

# 2. Aplicar o filtro anticausal (Hc(1/z))
h_cascade = iir.filtrev(b_causal, a_causal, h_causal)

# Plotar os resultados
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(n, h_causal, 'b', markerfmt='bo', basefmt=" ")
plt.title('Resposta ao Impulso do Filtro Causal (1ª Etapa)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n, h_cascade, 'r', markerfmt='ro', basefmt=" ")
plt.title('Resposta ao Impulso Total (Cascata)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

# Calcular e plotar a resposta em frequência
w, H = iir.resposta_em_frequencia_fft(h_cascade)
H_mag = np.abs(H)

plt.subplot(3, 1, 3)
plt.plot(w, H_mag)
plt.title('Magnitude da Resposta em Frequência (Cascata)')
plt.xlabel('Frequência [rad]')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()