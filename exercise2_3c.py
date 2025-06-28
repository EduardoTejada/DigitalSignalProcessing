import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk

# Coeficientes do filtro causal H1(z)
b1 = [0.1149, 0.0596, -0.0416]
a1 = [1, -1.2062, 0.5406]

# Calcular polos e zeros
zeros, poles, _ = tf2zpk(b1, a1)

# Plotar diagrama de polos e zeros
plt.figure(figsize=(8, 6))

# Círculo unitário
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)

# Polos e zeros
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='b', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Polos')

plt.title('Diagrama de Polos e Zeros')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# Calcular atraso de grupo teórico
w = np.linspace(0, np.pi, 500)
_, H1 = freqz(b1, a1, worN=w)
gd1 = -np.diff(np.unwrap(np.angle(H1)))
gd1 = np.append(gd1, gd1[-1])  # Extender o último valor

plt.figure(figsize=(12, 4))
plt.plot(w, gd1)
plt.title('Atraso de Grupo Teórico da Parte Causal')
plt.xlabel('Frequência [rad]')
plt.ylabel('Amostras')
plt.grid(True)
plt.show()
