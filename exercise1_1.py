import numpy as np
import matplotlib.pyplot as plt

def gdel(x, n, Lfft):
    """
    Calcula o atraso de grupo de x[n].
    x: Sinal
    n: Vetor de índices de tempo
    Lfft: Tamanho da FFT
    """
    X = np.fft.fft(x, Lfft)
    dXdw = np.fft.fft(n * x, Lfft)  # Transformada de n*x[n]
    gd = np.fft.fftshift(np.real(dXdw / X))  # Atraso de grupo
    w = (2 * np.pi / Lfft) * np.arange(Lfft) - np.pi  # Frequências normalizadas
    return gd, w

# Sinal simétrico
nn = np.arange(8)  # 0 <= n <= 7
x = np.array([1, 2, 3, 4, 4, 3, 2, 1])

# Impulso em n=5
nn = np.arange(-64, 64)  # Vetor de tempo
n0 = 5
x = (nn == n0).astype(float)

# Calcula o atraso de grupo
gd, w = gdel(x, nn, Lfft=1024)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(w, gd)
plt.axhline(y=3.5, color='r', linestyle='--')
plt.xlabel("Frequência [rad/sample]")
plt.ylabel("Atraso de Grupo (amostras)")
plt.ylim(0, 5); plt.grid(); plt.legend()
plt.show()