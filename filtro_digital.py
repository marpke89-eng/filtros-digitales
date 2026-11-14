import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==============================================================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ==============================================================================
fs = 1000.0  # Frecuencia de muestreo en Hz
T = 1.0      # Duración de la señal en segundos
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Componentes de frecuencia: baja (10 Hz), media (60 Hz) y alta (150 Hz)
f_baja = 10.0
f_media = 60.0
f_alta = 150.0

senal_limpia = (1.0 * np.sin(2 * np.pi * f_baja * t) +
                0.8 * np.sin(2 * np.pi * f_media * t) +
                0.6 * np.sin(2 * np.pi * f_alta * t))

# Agregar ruido blanco gaussiano
np.random.seed(42)
ruido = 0.5 * np.random.normal(size=t.shape)
senal_ruidosa = senal_limpia + ruido

# Función auxiliar para calcular la Transformada Rápida de Fourier (FFT)
def calcular_fft(senal, fs):
    N = len(senal)
    yf = np.fft.rfft(senal)
    xf = np.fft.rfftfreq(N, 1 / fs)
    magnitud = (2.0 / N) * np.abs(yf)
    return xf, magnitud

# ==============================================================================
# 2. DISEÑO DE LOS FILTROS DIGITALES (IIR Y FIR)
# ==============================================================================
nyquist = 0.5 * fs

# A) Filtro IIR Pasa Bajos (Butterworth, orden 4, corte en 25 Hz -> aísla 10 Hz)
fc_pb = 25.0
b_pb, a_pb = signal.butter(N=4, Wn=fc_pb / nyquist, btype='low')

# B) Filtro IIR Pasa Altos (Butterworth, orden 4, corte en 100 Hz -> aísla 150 Hz)
fc_pa = 100.0
b_pa, a_pa = signal.butter(N=4, Wn=fc_pa / nyquist, btype='high')

# C) Filtro IIR Pasa Bandas (Butterworth, orden 4, banda 40-80 Hz -> aísla 60 Hz)
fc_pbanda = [40.0, 80.0]
b_pbanda, a_pbanda = signal.butter(N=4, Wn=[f / nyquist for f in fc_pbanda], btype='band')

# D) Filtro FIR Pasa Bajos de comparación (Ventana Hamming, 101 coeficientes, corte 25 Hz)
numtaps = 101
b_fir_pb = signal.firwin(numtaps=numtaps, cutoff=fc_pb / nyquist, window='hamming')
a_fir_pb = [1.0]

# ==============================================================================
# 3. APLICACIÓN DE LOS FILTROS A LA SEÑAL
# ==============================================================================
# Se usa filtfilt para filtrado de fase cero (evita desfase temporal)
senal_pb = signal.filtfilt(b_pb, a_pb, senal_ruidosa)
senal_pa = signal.filtfilt(b_pa, a_pa, senal_ruidosa)
senal_pbanda = signal.filtfilt(b_pbanda, a_pbanda, senal_ruidosa)
senal_fir_pb = signal.filtfilt(b_fir_pb, a_fir_pb, senal_ruidosa)

# ==============================================================================
# 4. VISUALIZACIÓN DE RESULTADOS
# ==============================================================================
plt.style.use('seaborn-v0_8-darkgrid')

# --- Figura 1: Respuesta en Frecuencia de los Filtros ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
for b, a, label in [(b_pb, a_pb, 'IIR Pasa Bajos (25 Hz)'),
                    (b_pa, a_pa, 'IIR Pasa Altos (100 Hz)'),
                    (b_pbanda, a_pbanda, 'IIR Pasa Bandas (40-80 Hz)'),
                    (b_fir_pb, a_fir_pb, 'FIR Pasa Bajos (Hamming, 25 Hz)')]:
    w, h = signal.freqz(b, a, worN=2000, fs=fs)
    ax1.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-5)), label=label)

ax1.set_title('Respuesta en Frecuencia de los Filtros Digitales')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.set_ylabel('Ganancia [dB]')
ax1.set_xlim(0, 250)
ax1.set_ylim(-60, 5)
ax1.legend()
plt.tight_layout()
plt.savefig('respuesta_frecuencia.png', dpi=300)

# --- Figura 2: Dominio del Tiempo (Primeros 0.3 segundos para claridad) ---
fig2, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
muestras = int(0.3 * fs)

axs[0].plot(t[:muestras], senal_ruidosa[:muestras], color='gray', alpha=0.8, label='Señal con Ruido')
axs[0].plot(t[:muestras], senal_limpia[:muestras], color='black', linestyle='--', label='Señal Original Limpia')
axs[0].set_title('Señal de Entrada (10 Hz + 60 Hz + 150 Hz + Ruido Blanco)')
axs[0].legend(loc='upper right')

axs[1].plot(t[:muestras], senal_pb[:muestras], color='tab:blue')
axs[1].set_title('Salida Filtro Pasa Bajos (< 25 Hz) -> Componente de 10 Hz aislada')

axs[2].plot(t[:muestras], senal_pbanda[:muestras], color='tab:green')
axs[2].set_title('Salida Filtro Pasa Bandas (40-80 Hz) -> Componente de 60 Hz aislada')

axs[3].plot(t[:muestras], senal_pa[:muestras], color='tab:red')
axs[3].set_title('Salida Filtro Pasa Altos (> 100 Hz) -> Componente de 150 Hz aislada')
axs[3].set_xlabel('Tiempo [s]')

plt.tight_layout()
plt.savefig('dominio_tiempo.png', dpi=300)

# --- Figura 3: Dominio de la Frecuencia (Espectro FFT) ---
fig3, axs_f = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

espectros = [
    (senal_ruidosa, 'Espectro Señal Original con Ruido', 'gray'),
    (senal_pb, 'Espectro Pasa Bajos (Pico en 10 Hz)', 'tab:blue'),
    (senal_pbanda, 'Espectro Pasa Bandas (Pico en 60 Hz)', 'tab:green'),
    (senal_pa, 'Espectro Pasa Altos (Pico en 150 Hz)', 'tab:red')
]

for ax, (sig, titulo, color) in zip(axs_f, espectros):
    xf, mag = calcular_fft(sig, fs)
    ax.plot(xf, mag, color=color)
    ax.set_title(titulo)
    ax.set_ylabel('Amplitud')
    ax.set_xlim(0, 200)

axs_f[-1].set_xlabel('Frecuencia [Hz]')
plt.tight_layout()
plt.savefig('dominio_frecuencia.png', dpi=300)
plt.show()
