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
h_cascade = iir.anticausal_filter([1], [1, -a], h_causal)

# Plotar os resultados
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.stem(n, h_causal, 'b', markerfmt='bo', basefmt=" ")
plt.title('Resposta ao Impulso do Filtro Causal (1ª Etapa)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(n, h_cascade, 'r', markerfmt='ro', basefmt=" ")
plt.title('Resposta ao Impulso Total (Cascata Causal + Anticausal)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.xlim(-30, 30)
plt.grid(True)

plt.tight_layout()
plt.show()