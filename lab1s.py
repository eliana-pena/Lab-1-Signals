# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:32:50 2024

@author: Estudiante
"""
#importar paquete wfdb para leer "records" de physionet 
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# cargar la información (archivos .dat y .hea)
signal = wfdb.rdrecord('cu05')

# obtener valores de y en la señal 
valores = signal.p_signal

# Aplanar los valores a una dimensión
valores = valores.flatten()

# Eliminar valores NaN
valores_limpios = [x for x in valores if not np.isnan(x)]

# Función para calcular la media de los valores
def calc_media(valores):
    total=0
    for x in valores:
        total += x
    return total / len(valores)

# Función para calcular la mediana de los valores
# def calc_mediana(valores):
#    val_ord = sorted(valores)
#    n = len(val_ord)
#    mid = n // 2
#    if n % 2 == 0:
#        return (val_ord[mid - 1] + val_ord[mid]) / 2
#    else:
#        return val_ord[mid]

# Función para calcular la desviación estándar de los valores
def calc_desv(valores, media):
    suma_difcuad = 0
    for x in valores:
        suma_difcuad += (x - media) ** 2 #Sumando en un bucle
    return (suma_difcuad / len(valores)) ** 0.5

# Función para calcular el coeficiente de variación
def calc_coe(media,desviacion):
    return desviacion/media

# Función para calcular la relación señal-ruido (SNR)
def snr (valores, ruido):
    fsum = 0
    ssum = 0
    potv= 0
    potr = 0
    for x in valores:
        fsum += (x) ** 2
    potv = fsum/len(valores)
    
    for x in ruido:
        ssum += (x) ** 2
    potr = ssum/len(ruido)

    return 10 * np.log10(potv / potr)

# Media, mediana, desviación estándar - calculadas por funciones de numpy
media = np.nanmean(valores)
#mediana = np.nanmedian(valores)
desviacion = np.nanstd(valores)

# Media, mediana, desviación estándar - cálculos con fórmulas
mediac = calc_media(valores_limpios)
#medianac = calc_mediana(valores_limpios)
desvc = calc_desv(valores_limpios, mediac)
coef = calc_coe(media, desviacion)

print(f"Media: {media}, Media*: {mediac}")
#print(f"Mediana: {mediana}, Mediana*: {medianac}")
print(f"Desviación: {desviacion}, Desviación*: {desvc}")
print(f"Coeficiente de desviacion: {coef}")

#plt.plot(valores)
#plt.title('Señal')
#plt.xlabel('Muestras')
#plt.ylabel('Valor de la señal')
#plt.show()

# Calcular histograma manualmente
min_val=min(valores_limpios)
max_val=max(valores_limpios)

nbin = 20
wbin= (max_val-min_val)/nbin

bins = [0] * nbin

# Asignar valores a bins 
for val in valores_limpios:
    binx = int((val - min_val) / wbin)
    if binx == nbin:
        binx -= 1
    bins[binx] +=1

# Crear bordes de bins para el histograma
ebin=[min_val + i * wbin for i in range (nbin +1)]

# Graficar histograma
plt.bar(ebin[:-1], bins, width=wbin, edgecolor='black') # funcion para graficar diagrama de barras
plt.xlabel('Valor de la señal')
plt.ylabel('Frecuencia')
plt.title('Histograma de la señal')
plt.show()

#plt.hist(valores,bins=20)
#plt.title('Histograma de la señal')
#plt.xlabel('Muestras')
#plt.ylabel('frecuencia')
#plt.show()

nm=len(valores_limpios)

##################################################################
# Ruido Gaussiano
medrg=0
desvrg=1
rgauss=np.random.normal(medrg,desvrg,nm)

ampmax = max(abs(min_val),abs(max_val))- 3
rgaussn = rgauss/np.max(np.abs(rgauss))*ampmax

# Infectar la señal con ruido Gaussiano
sginf = valores_limpios + rgaussn 

plt.figure(figsize=(15,5))

plt.subplot(3,1,1)
plt.plot(valores_limpios)
plt.title('Señal origianl')

plt.subplot(3,1,2)
plt.plot(rgaussn,color = 'red')
plt.title('Ruido Gaussiano Normalizado')

plt.subplot(3,1,3)
plt.plot(sginf,color = 'purple')
plt.title('Señal Infectada con Ruido Gaussiano')

plt.tight_layout()
plt.show()
##################################################################
# Ruido de Pulso
numpul=10
rpul=np.zeros(nm)
indpul= np.random.choice(nm,numpul,replace=False)
rpul[indpul] = np.random.choice([-5,5],numpul)

# Infectar la señal con ruido de pulso unitario
spinf= valores_limpios + rpul

plt.figure(figsize=(15, 5))

plt.subplot(3, 1, 1)
plt.plot(valores_limpios, color='blue')
plt.title('Señal Original')

plt.subplot(3, 1, 2)
plt.plot(rpul, color='yellow')
plt.title('Ruido de Pulso Unitario')

plt.subplot(3, 1, 3)
plt.plot(spinf, color='green')
plt.title('Señal Infectada con Ruido de Pulso Unitario')

plt.tight_layout()
plt.show()

###################################################################
# Ruido de Tipo Artefacto

rart = np.zeros(nm)
ampart = 5
durart = 5000
artini = np.random.randint(0,nm)
rart[artini: artini + durart]= ampart * np.sin(np.linspace(0,2*np.pi,durart))

# Infectar la señal con ruido de tipo artefacto
sainf=valores_limpios + rart

plt.figure(figsize=(15, 5))

plt.subplot(3, 1, 1)
plt.plot(valores_limpios, color='blue')
plt.title('Señal Original')

plt.subplot(3, 1, 2)
plt.plot(rart, color='red')
plt.title('Ruido de Tipo Artefacto')

plt.subplot(3, 1, 3)
plt.plot(sainf, color='green')
plt.title('Señal Infectada con Ruido de Tipo Artefacto')

plt.tight_layout()
plt.show()

##################################################################

# Calcular y mostrar la SNR para cada tipo de ruido
snr_valg = snr(valores_limpios,rgaussn)
snr_valp = snr(valores_limpios,rpul)
snr_vala = snr(valores_limpios,rart)
print(f"Relacion señal ruido Gauss: {snr_valg} dB")
print(f"Relacion señal ruido Impulso: {snr_valp} dB")
print(f"Relacion señal ruido Artefacto: {snr_vala} dB")







