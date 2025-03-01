# Problema-del-coctel
## Introducción 
A continuación encontrará los pasos necesarios para realizar un código en pyhton que permita concentrarse en una sola fuente sonora dentro de un entorno con variedad de emisores de sonido haciendo uso del algoritmo de Análisis de Componentes Independientes (ICA) para separar fuentes sonoras. Para lo anterior se realizó la grabación de dos audios al mismo tiempo, tomando un ruido blanco y la separación entre cada micrófono presente en la sala, de esta manera se plasmarán los resultados obtenidos y la explicación detallada del código, brindando las herramientas para su utilización en situaciones donde hay interferencias acústicas y se desea aislar una señal específica, como en el caso del desarrollo de dispositivos auditivos.

## Resultados
Primero se ubicaron dos micrófonos de manera estratégica en el espacio de grabación de tal manera que ambos lograron captar señales provenientes de las dos fuentes de información (personas). Posterior a esto se inicializó la grabación, adquiriendo las señales que se van a analizar. Además, se registró el ruido blanco presente en la sala de filmación. Una vez realizado lo anteriormente mencionado se obtuvo el análisis temporal representado en las siguientes gráficas:
<div align="center">
  <img src="https://github.com/user-attachments/assets/4f4eeffa-ad47-4814-9b5c-363742495334" width="500" height="200">
</div>
De acuerdo con las gráficas correspondientes al análisis temporal se pudo apreciar que ambas señales cuentan con una forma muy similar, debido a que la fuente de información sonora captada era la misma para ambos casos. También se observó que existe un leve desfase entre ambas señales debido a que las fuentes sonoras no se encontraban a la misma distancia con ambos micrófonos.   

 Después se realizó el análisis espectral por medio de la Transformada Rápida de Fourier (FFT), adquiriendo las siguientes gráficas:
<div align="center">
  <img src="https://github.com/user-attachments/assets/6b386a65-8cd1-4dc5-b379-52ceb1e8a784" width="500" height="200">
</div>
El espectro de frecuencia muestra que la mayor parte de la energía de la señal está en bajas frecuencias características en la voz humana y algunos ruidos de fondo como el tráfico. Además se observa una caída en la amplitud a partir de los 5 kHz indicando que la señal no tiene muchas componentes agudas.  

Adicional a esto se obtuvo el espectograma de ambos micrófonos, el cual permite diferenciar en qué momentos y a qué frecuencias aparecen distintas voces, y en qué momentos una persona hablaba más fuerte o habían pausas en la conversación.

<div align="center">
  <img src="https://github.com/user-attachments/assets/70d6543e-1d8d-4e45-9066-eb16b0af05f2" width="500" height="200">
</div>

Por último se obtuvo la separación de las voces para ambos casos, encontrando el SNR de la señal original con la separada y de la señal original con el ruido blanco. A continuación se presetan las gráficas y los datos obtenidos:
<div align="center">
  <img src="https://github.com/user-attachments/assets/2fb5f950-f273-4267-9310-54e80ce07041" width="500" height="200">
</div>

SNR entre Mic 1 y su fuente separada: 37.92 dB
SNR entre Mic 2 y su fuente separada: -3.58 dB
SNR entre Voz 1 y Ruido: -1.43 dB
SNR entre Voz 2 y Ruido: -1.04 dB

De acuerdo con los resultados obtenidos del SNR se infiere que la separación de la señal en el micrófono 1 es muy efectiva. Un SNR alto sugiere que la fuente ha sido correctamente aislada con poco ruido residual.    

Para el caso del micrófono 2 el ruido es más fuerte que la señal de interés, lo que indica que la separación no fue efectiva en este caso. En el SNR de las voces tenemos que la voz es apenas más débil que el ruido, lo que indica que el entorno es muy ruidoso y que la separación de fuentes no fue del todo efectiva. En la voz 2 se presenta un escenario similar al caso anterior, en el cual el ruido sigue estando al mismo nivel que la señal de voz, lo que puede dificultar la nitidez de la información.

AUDIOS SEPARADOS:

Voz 1:
https://github.com/user-attachments/assets/8998c038-7742-4589-9cdc-62049e93663d

Voz 2:
https://github.com/user-attachments/assets/2c69e7fc-2665-4dd8-a64f-f65cf3a9f647



1.	¿Cómo afecta la posición relativa de los micrófonos y las fuentes sonoras en la efectividad de la separación de señales?
La posición relativa de los micrófonos afecta dado que al no haber una distancia adecuada las frecuencias cambian, lo que quiere decir que en caso de que los micrófonos queden muy cercanos unos a otros se tendrán frecuencias más parecidas y menos diversidad en la señal, lo que generará mucho más difícil el proceso de separar los audios; en cambio si los micrófonos se encuentran a una distancia prudente entre sí las frecuencias serán más distintas con mayor diversidad y por lo tanto facilitara y volverá más preciso el método de ICA. Cabe recalcar que la orientación de los micrófonos también es importante dado que, si se orientan en direcciones diferentes, se aumenta la diversidad de las señales y mejora la precisión de la separación.

2.	¿Qué mejoras implementaría en la metodología para obtener mejores resultados?
Dado que en este laboratorio los audios no fueron tomados de manera óptima, sería necesario aumentar la distancia entre los micrófonos y mejorar la precisión en su posición y orientación. Además, se debería elegir un lugar de grabación con menos ruidos externos y eco, ya que estos factores generan distorsión en las señales. Finalmente, se recomienda utilizar voces con mayores diferencias de frecuencia para que las señales sean más fáciles de distinguir y así lograr una separación más efectiva de los audios mediante el ICA.

## Instrucciones
1. Se importan las librerías numpy para las operaciones matemáticas, matplot para los gráficos, scipy.io,wavfile para leer y escribir los archivos WAP (audios), spicy.signal para calcular y visualizar el espectro de las señales y sklearn para aplicar el algoritmo de separación de fuentes mediante ICA. 
Posteriormente Se cargan los archivos de audio, buscando la ruta desde el computador, también lo puede hacer subiendo los archivos a la libreria de python directamente

```python
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import FastICA

fs1, audio1 = wav.read(r"C:\Users\Camila Martinez\Downloads\mic1.wav")
fs2, audio2 = wav.read(r"C:\Users\Camila Martinez\Downloads\mic2.wav")
fs3, ruido = wav.read(r"C:\Users\Camila Martinez\Downloads\rudsala.wav")
```

2. A continuación se verifica que las señales de audio tengan la misma frecuencia de muestreo mediante un if, en caso de no tenerlas se lanza un error. 
Seguidamente para evitar problemas al procesar las señales, se recortan a la misma longitud, en este caso la longitud mínima de las tres señales.

``` python
if fs1 != fs2 or fs1 != fs3:
    raise ValueError("Las señales tienen diferentes frecuencias de muestreo.")

min_len = min(len(audio1), len(audio2), len(ruido))
audio1, audio2, ruido = audio1[:min_len], audio2[:min_len], ruido[:min_len]
```
3. A continuación se normalizan las señales de audio para evitar que se saturen al procesarlas, para esto se divide por el valor absoluto máximo de cada señal.
Finalmente se construye una matriz x donde cada columna hace referencia a una de las señales captadas por los micrófonos.
``` python
audio1 = audio1 / np.max(np.abs(audio1))
audio2 = audio2 / np.max(np.abs(audio2))
ruido = ruido / np.max(np.abs(ruido))

X = np.c_[audio1, audio2]
```

#### Análisis temporal
4. Se grafican las señales de los dos micrófonos para visualizarlas en el dominio del tiempo.

``` Python
plt.figure(figsize=(12, 4))
for i in range(2):
    plt.subplot(2, 1, i+1)
    plt.plot(X[:, i])
    plt.title(f"Señal en el tiempo - Micrófono {i+1}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()
``` 

#### Análisis spectral
5. Se realiza una función para calcular la Transformada Rápida de Fourier con la finalidad de analizar su espectro de frecuencia, posteriormente se visualiza mediante una gráfica
``` python
def calcular_fft(signal, fs):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_values = np.abs(np.fft.fft(signal)) / N
    return freqs[:N//2], fft_values[:N//2]  # Solo parte positive

plt.figure(figsize=(12, 4))
for i in range(2):
    freqs, fft_values = calcular_fft(X[:, i], fs1)
    plt.subplot(2, 1, i+1)
    plt.plot(freqs, fft_values)
    plt.title(f"Espectro de Frecuencia - Micrófono {i+1}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()
```

#### Análisis del espectrograma
6. Se calcula el espectrograma de las señales de los micrófonos, lo que permite observar cómo varían las frecuencias a lo largo del tiempo.
``` python
plt.figure(figsize=(12, 4))
for i in range(2):
    f, t, Sxx = spectrogram(X[:, i], fs1)
    plt.subplot(2, 1, i+1)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel(f"Frecuencia (Hz)")
    plt.colorbar(label="dB")
    plt.title(f"Espectrograma - Micrófono {i+1}")
plt.xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()
```
#### Separación de fuentes
7. Se aplica la técnica de ICA para separar los audios, posteriormente se renormaliza la señal y por ultimo se guardan las señales separadas en formato WAV

``` python
 ica = FastICA(n_components=2)
separated_sources = ica.fit_transform(X)

for i in range(2):
    separated_sources[:, i] = separated_sources[:, i] / np.max(np.abs(separated_sources[:, i]))  # Re-normalizar

for i in range(2):
    wav.write(f"fuente_separada_{i+1}.wav", fs1, (separated_sources[:, i] * 32767).astype(np.int16))
```
#### SNR
8. Se define la funión para sacar el SNR entre la señal original y la señal recuperada, posteriormente se calcula 
 
``` python
def snr(original, recovered):
    noise = original - recovered
    return 10 * np.log10(np.sum(original*2) / np.sum(noise*2))
```
9. Se calcula el SNR entre las señales obtenidas con las recuperadas y posteriormente el SNR entre la señal original y el ruido.
Se imprimen los valores en consola.
``` python
snr_values = [snr(X[:, i], separated_sources[:, i]) for i in range(2)]

snr_audio1_ruido = snr(audio1, ruido)
snr_audio2_ruido = snr(audio2, ruido)

for i, snr_val in enumerate(snr_values):
    print(f"SNR entre Mic {i+1} y su fuente separada: {snr_val:.2f} dB")
print(f"SNR entre Voz1 y Ruido: {snr_audio1_ruido:.2f} dB")
print(f"SNR entre Voz2 y Ruido: {snr_audio2_ruido:.2f} dB")
```

#### Comparación de las señales
10. Se grafican las señales originales y las señales separadas para poder ser compradas y analizadas. 
```python
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 2, 2*i+1)
    plt.plot(X[:, i])
    plt.title(f"Señal capturada Mic {i+1}")

    plt.subplot(2, 2, 2*i+2)
    plt.plot(separated_sources[:, i])
    plt.title(f"Fuente separada {i+1}")

plt.tight_layout()
plt.show()
```




## Requerimientos
1. Python 3.9
2. Librerias Numpy, matplotlib, scipy.io y sklearn 
3. Mínimo dos audios con diferentes voces
4. Audio de ruido blanco 
5. Distancia entre los microfonos



## Uso
 Problema del coctel por Camila Martínez y Paula Vega  
 Publicado 28/02/25

 ## Referencias
1. ¿Cómo concatenar varios audios utilizando Python? (s. f.). Stack Overflow En Español. https://es.stackoverflow.com/questions/226314/c%C3%B3mo-concatenar-varios-audios-utilizando-python
2. Programacionpython. (2022, 25 abril). MANIPULANDO AUDIOS EN PYTHON, CON «pydub». El Programador Chapuzas. https://programacionpython80889555.wordpress.com/2020/02/25/manipulando-audios-en-python-con-pydub/
3. Jonschkowski, R., & Kohl, J. (2023). Nonlinear Independent Component Analysis for Principled Disentanglement in Unsupervised Deep Learning. arXiv. https://arxiv.org/abs/2303.16535
4. SimpleScience. (2023, marzo 31). Retos y Soluciones en el Análisis de Componentes Independientes. SimpleScience AI. https://simplescience.ai/es/2023-03-31-retos-y-soluciones-en-el-analisis-de-componentes-independientes--anzov7
