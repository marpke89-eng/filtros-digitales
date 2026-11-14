# Implementación y Evaluación de Filtros Digitales en Python

Este repositorio contiene el desarrollo de la **Actividad Formativa 3**, enfocada en el diseño, simulación y análisis de filtros digitales (pasa bajos, pasa altos y pasa bandas) aplicados a señales contaminadas con ruido blanco.

## Requisitos
* Python 3.8+
* NumPy (`pip install numpy`)
* SciPy (`pip install scipy`)
* Matplotlib (`pip install matplotlib`)

## Estructura del Proyecto
* `filtros_digitales.py`: Código principal con generación de señal, diseño de filtros IIR (Butterworth) y FIR (Hamming), filtrado de fase cero y generación de gráficas.
* `respuesta_frecuencia.png`: Curvas de ganancia (dB) vs. frecuencia de los filtros diseñados.
* `dominio_tiempo.png`: Comparativa temporal antes y después del filtrado.
* `dominio_frecuencia.png`: Espectros FFT demostrando la atenuación de frecuencias no deseadas.
