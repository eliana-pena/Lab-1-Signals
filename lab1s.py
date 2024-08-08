# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 17:02:21 2024

@author: 57322
"""

#importa las librerias
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import random



#obtener los valores de Y de la señal
signal = wfdb.rdrecord('Subject10_AccTempEDA')
valores = signal.p_signal[:,1]
longitud =len(valores) #numero de muestras 

# Graficar la señal
plt.plot(valores)
plt.title("Señal Fisiológica")
plt.xlabel("Tiempo (s)")
plt.ylabel("EDA (uS)")
plt.show()

"ESTADÍSTICOS DESCRIPTIVOS"

#Estadísticos descriptivos:Media manual
val_total= 0
for i in valores:
    val_total+=i

media_manual = (val_total/longitud)

#Cálculo desviación estándar a partir de operaciones 
  
diferencia_cuadrados = ((i-media_manual)**2 for i in valores)
suma_diferencia= 0 
for j in diferencia_cuadrados:
    suma_diferencia+=j

    
desviacion_estandar_manual = (float(suma_diferencia/(longitud -1 )))**0.5
coeficiente_variacion_manual = (desviacion_estandar_manual / media_manual)

print(f"Media calculada manualmente: {media_manual}")
print(f"Desviación estándar calculada manualmente: {desviacion_estandar_manual}")
print(f"Coeficiente de variación calculado manualmente: {coeficiente_variacion_manual:.2f}%")

#Estadísticos descriptivos: Media y Desviación Estándar usando funciones predefinidas

media_predefinida = np.mean(valores)
desviacion_estandar_predefinida = np.std(valores, ddof=1)
coeficiente_variacion_predefinida = (desviacion_estandar_predefinida / media_predefinida) 

print(f"Media usando funciones predefinidas: {media_predefinida}")
print(f"Desviación estándar usando funciones predefinidas: {desviacion_estandar_predefinida}")
print(f"Coeficiente de variación usando funciones predefinidas: {coeficiente_variacion_predefinida:.2f}%")

#Graficar el histograma de la señal mediante funciones de python
plt.hist(valores, bins=8,edgecolor='black')
plt.title("Histograma de la señal con funciones")
plt.xlabel("Valor (uS)")
plt.ylabel("# muestra")
plt.show()

#Histograma de la señal manual

num_bins = 8 
min_val = valores[0]
max_val = valores[0]
for valor in valores:
    if valor < min_val:
        min_val = valor
    if valor > max_val:
        max_val = valor

bin_ancho = (max_val - min_val) / num_bins
bins = [0] * num_bins

for valor in valores:
    bin_cajita = int((valor - min_val) / bin_ancho)
    if bin_cajita == num_bins:  
        bin_cajita-= 1
    bins[bin_cajita] += 1

# Graficar el histograma manual
bin_borde = [min_val + i * bin_ancho for i in range(num_bins + 1)]
plt.bar(bin_borde[:-1], bins, width=bin_ancho, edgecolor='black')
plt.title("Histograma de la señal (manual)")
plt.xlabel("Valor (uS)")
plt.ylabel("# muestra")
plt.show()

# Generar ruido gaussiano
ruido_gaussiano = np.random.normal(0, np.std(valores), len(valores))
signal_ruido_gaussiano = valores + ruido_gaussiano


potencia_senal = np.mean(valores ** 2)
potencia_ruido_gaussiano = np.mean(ruido_gaussiano ** 2)
print(potencia_ruido_gaussiano,potencia_senal)

snr_gaussiano = 10 * np.log10(potencia_senal / potencia_ruido_gaussiano)
print(f"SNR con ruido gaussiano: {snr_gaussiano:.2f} dB")

# Graficar la señal contaminada
plt.plot(signal_ruido_gaussiano)
plt.title("Señal con Ruido Gaussiano")
plt.xlabel("Tiempo (s)")
plt.ylabel("EDA (uS)")
plt.show()


# Generar ruido de impulso
ruido_impulso = np.zeros(len(valores))
num_impulsos = int(0.01 * len(valores))  
impulso_amplitud = np.max(np.abs(valores)) * 2  # amplitud de los impulsos

for _ in range(num_impulsos):
    posicion = random.randint(0, len(valores) - 1)
    ruido_impulso[posicion] = impulso_amplitud * (1 if random.random() < 0.5 else -1)


signal_ruido_impulso = valores + ruido_impulso

potencia_ruido_impulso = np.mean(ruido_impulso ** 2)

# Calcular el SNR
snr_impulso = 10 * np.log10(potencia_senal / potencia_ruido_impulso)
print(f"SNR con ruido de impulso: {snr_impulso:.2f} dB")

plt.plot(signal_ruido_impulso)
plt.title("Señal con Ruido de Impulso")
plt.xlabel("Tiempo (s)")
plt.ylabel("EDA (uS)")
plt.show()

# Generar ruido tipo artefacto
ruido_artefacto = np.zeros(len(valores))
num_artefactos = int(0.01 * len(valores))  
artefacto_amplitud = np.max(np.abs(valores)) * 2  
artefacto_duracion = 10 

for _ in range(num_artefactos):
    posicion_inicio = random.randint(0, len(valores) - artefacto_duracion)
    ruido_artefacto[posicion_inicio:posicion_inicio + artefacto_duracion] = artefacto_amplitud


signal_ruido_artefacto = valores + ruido_artefacto
potencia_ruido_artefacto = np.mean(ruido_artefacto ** 2)

# Calcular el SNR
snr_artefacto = 10 * np.log10(potencia_senal / potencia_ruido_artefacto)
print(f"SNR con ruido tipo artefacto: {snr_artefacto:.2f} dB")

# Graficar la señal contaminada
plt.plot(signal_ruido_artefacto)
plt.title("Señal con Ruido Tipo Artefacto")
plt.xlabel("Tiempo (s)")
plt.ylabel("EDA (uS)")
plt.show()







