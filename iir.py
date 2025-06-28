import numpy as np
from scipy import signal

def gdel(x, n, Lfft=1024):
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

def resposta_em_frequencia(b, a, worN=1024, whole=False):
    # Resposta em frequência (EXATA: usando freqz)
    w, H_exato = signal.freqz(b, a, worN, whole)
    return w, H_exato

def resposta_em_frequencia_fft(h, Lfft=1024, n=64):
    # nn = np.arange(-n, n)
    # impulse = (nn == 0).astype(float)
    # hc = signal.lfilter(b, a, impulse)
    # Resposta em frequência (APROXIMADA: usando FFT da hc[n])
    H_approx = np.fft.fft(h, Lfft)  # Usamos metade por simetria
    w_approx = np.linspace(0, 2*np.pi, Lfft, endpoint=False)
    return w_approx, H_approx

def resposta_ao_impulso_causal(b, a, n=64):
    nn = np.arange(-n, n)  # Vetor de tempo
    impulse = (nn == 0).astype(float)  # Impulso em n=0
    return signal.lfilter(b, a, impulse)  # Resposta ao impulso causal

def resposta_ao_impulso_anticausal(b, a, n=64):
    nn = np.arange(-n, n)  # Vetor de tempo
    impulse = (nn == 0).astype(float)  # Impulso em n=0
    return np.flip(signal.lfilter(b, a, np.flip(impulse)))  # Resposta ao impulso causal

def filtrev(b, a, x):
    """Implementa um filtro anticausal IIR usando reversão temporal"""
    # Implementação do filtro anticausal (3 passos):
    # 1. Reverte o sinal de entrada: x[-n]
    # 2. Filtra causalmente com Hc(z)
    # 3. Reverte a saída: y[n] = y_causal[-n]
    y = np.flip(signal.lfilter(b, a, np.flip(x)))
    return y

def causal_filter(b, a, x):
    """Filtro IIR causal"""
    return signal.lfilter(b, a, x)

def zero_phase_cascade(b, a, x):
    """Implementação em cascata do filtro de fase zero"""
    # Filtro causal
    y = causal_filter(b, a, x)
    # Filtro anticausal
    return filtrev(b, a, y)

def zero_phase_parallel(b1, a1, x):
    """Implementação em paralelo do filtro de fase zero"""
    # Parte causal
    y1 = causal_filter(b1, a1, x)
    # Parte anticausal
    y2 = filtrev(b1, a1, x)
    return y1 + y2