import numpy as np
import matplotlib.pyplot as plt
import iir

# Coeficientes para implementação em cascata (usando Hc(z) do Exercício 2.3a)
b_cascade = [0.1432, 0.0117, 0.1432]
a_cascade = [1, -1.2062, 0.5406]

# Coeficientes para implementação em paralelo (do Exercício 2.3b)
b_parallel = [0.1149, 0.0596, -0.0416]
a_parallel = [1, -1.2062, 0.5406]

# Criar o sinal de onda quadrada
n = np.arange(-64, 64)  # -64 <= n <= 63
x = np.zeros_like(n, dtype=float)
x[(n >= -22) & (n <= -8)] = -2
x[(n >= -7) & (n <= 7)] = 2
x[(n >= 8) & (n <= 22)] = -3

# Processar o sinal
y_cascade = iir.zero_phase_cascade(b_cascade, a_cascade, x)
y_parallel = iir.zero_phase_parallel(b_parallel, a_parallel, x)

# Plotar os resultados
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(n, x, 'b', markerfmt='bo', basefmt=" ", label='Entrada')
plt.title('Sinal de Entrada (Onda Quadrada)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.stem(n, y_cascade, 'g', markerfmt='go', basefmt=" ", label='Cascata')
plt.title('Saída do Filtro de Fase Zero (Cascata)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.stem(n, y_parallel, 'r', markerfmt='ro', basefmt=" ", label='Paralela')
plt.title('Saída do Filtro de Fase Zero (Paralela)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Coeficientes do filtro causal Hc(z)
b_causal = [0.1432, 0.0117, 0.1432]
a_causal = [1, -1.2062, 0.5406]

# Processar com cascata de dois filtros causais (Hc(z)^2)
y_causal1 = iir.causal_filter(b_causal, a_causal, x)
y_causal = iir.causal_filter(b_causal, a_causal, y_causal1)

# Plotar comparação
plt.figure(figsize=(12, 6))

plt.stem(n, x, 'b', markerfmt='bo', basefmt=" ", label='Entrada')
plt.stem(n, y_causal, 'm', markerfmt='mo', basefmt=" ", label='Filtro Causal (Hc²)')
plt.title('Comparação: Filtro Causal vs. Entrada')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plotar comparação final
plt.figure(figsize=(12, 6))
plt.stem(n, y_parallel, 'g', markerfmt='go', basefmt=" ", label='Fase Zero (Paralela)')
plt.stem(n, y_causal, 'm', markerfmt='mo', basefmt=" ", label='Filtro Causal (Hc²)')
plt.title('Comparação Final: Fase Zero vs. Causal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()