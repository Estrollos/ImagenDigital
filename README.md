# Documentación: funciones de OpenCV utilizadas

Proyecto de agricultura de precisión: detección de zonas amarillentas y manchas en hojas con OpenCV clásico (sin IA, Machine Learning ni modelos externos).

## 1. Resumen del algoritmo

1. **Preprocesado:** se escala la imagen a 700 px de ancho y se suaviza.
2. **Separación hoja/fondo:** umbral de Otsu sobre la saturación, erosión, objeto mayor y dilatación.
3. **Clasificación por color:** en HSV se separan verde, amarillo y marrón. El corte verde/amarillo se calcula por hoja con el histograma del matiz.
4. **Medición:** porcentaje de píxeles amarillos y marrones sobre el área de la hoja, y severidad (leve, moderada o grave).
5. **Visualización:** contornos de las zonas afectadas y barras de ajuste en vivo.

## 2. Funciones de OpenCV

### Lectura, escalado y suavizado

| Función | Para qué sirve | Dónde se usa |
| --- | --- | --- |
| `cv2.imread(ruta)` | Carga una imagen en formato BGR. Devuelve `None` si la ruta es incorrecta | Programa principal |
| `cv2.resize(img, (ancho, alto), interpolation=metodo)` | Cambia el tamaño. `INTER_AREA` para reducir, `INTER_CUBIC` para aumentar | `escalar` |
| `cv2.blur(img, (k, k))` | Filtro de media (paso bajo). Elimina ruido y textura fina | `escalar` |

### Color y canales

| Función | Para qué sirve | Dónde se usa |
| --- | --- | --- |
| `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | Convierte de BGR a HSV: matiz (0-179), saturación y brillo (0-255) | `analizar` |
| `cv2.split(hsv)` | Separa la imagen en sus tres canales H, S y V | `separacion_hoja`, `clasificar` |
| `cv2.inRange(canal, min, max)` | Máscara binaria con los píxeles cuyo valor está dentro del rango | `clasificar` |

### Umbralización

| Función | Para qué sirve | Dónde se usa |
| --- | --- | --- |
| `cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` | Binariza con un umbral que calcula Otsu automáticamente, sin fijarlo a mano | `separacion_hoja` |
| `cv2.calcHist([h], [0], mascara, [180], [0, 180])` | Histograma del matiz, contando solo los píxeles de la máscara. Su máximo da el verde dominante de la hoja | `corte_verde_amarillo` |

### Operaciones lógicas y morfología

| Función | Para qué sirve | Dónde se usa |
| --- | --- | --- |
| `cv2.bitwise_and(a, b)` | Intersección de dos máscaras | `clasificar` |
| `cv2.bitwise_or(a, b)` | Unión de dos máscaras | `clasificar` |
| `cv2.bitwise_not(a)` | Invierte una máscara | `clasificar` |
| `cv2.erode(mascara, kernel)` | Encoge las zonas blancas. Separa objetos pegados, quita tallos finos y ruido | `separacion_hoja`, `clasificar` |
| `cv2.dilate(mascara, kernel)` | Agranda las zonas blancas. Se usa tras la erosión para recuperar el tamaño | `separacion_hoja`, `clasificar` |

Erosión seguida de dilatación elimina motas y líneas finas, como las venas claras, y deja las zonas compactas.

### Contornos y medidas

| Función | Para qué sirve | Dónde se usa |
| --- | --- | --- |
| `cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` | Encuentra los contornos exteriores de los objetos de una máscara | `mayor_contorno_relleno`, `dibujar_manchas`, `analizar` |
| `cv2.contourArea(contorno)` | Área de un contorno. Sirve para quedarse con el objeto mayor y descartar ruido | `mayor_contorno_relleno`, `dibujar_manchas` |
| `cv2.drawContours(img, contornos, -1, color, grosor)` | Dibuja contornos. Con `cv2.FILLED` los rellena | `mayor_contorno_relleno`, `dibujar_manchas`, `analizar` |
| `cv2.countNonZero(mascara)` | Cuenta los píxeles blancos. Se usa para calcular los porcentajes | `analizar` |

### Interfaz

| Función | Para qué sirve | Dónde se usa |
| --- | --- | --- |
| `cv2.putText(img, texto, pos, fuente, escala, color, grosor)` | Escribe texto sobre la imagen | `analizar` |
| `cv2.namedWindow`, `cv2.imshow` | Crean y muestran ventanas | Programa principal |
| `cv2.createTrackbar(nombre, ventana, valor, máximo, función)` | Barra deslizante que llama a una función al moverla. Permite ajustar parámetros en vivo | `poner_barras` |
| `cv2.waitKey(ms)` | Espera una tecla y refresca las ventanas | Programa principal |
| `cv2.destroyAllWindows()` | Cierra todas las ventanas | Programa principal |

## 3. Funciones de NumPy auxiliares

| Función | Uso |
| --- | --- |
| `np.ones((k, k), np.uint8)` | Crea el kernel de la erosión y la dilatación |
| `np.zeros_like(mascara)` | Crea una máscara vacía del mismo tamaño |
| `np.argmax(hist)` | Matiz más frecuente, es decir, el verde dominante |

## 4. Parámetros principales

| Parámetro | Valor | Significado |
| --- | --- | --- |
| `separar` | 7 | Tamaño de la erosión que aísla la hoja |
| `borde` | 7 | Píxeles del borde de la hoja que no se analizan |
| `h_marron_max` | 20 | Por debajo de este matiz, el píxel es marrón |
| `corte_fijo` | 30 | Corte verde/amarillo mínimo |
| `margen_verde` | 6 | Grados bajo el verde dominante a partir de los cuales es amarillento |
| `v_min_color` | 110 | Brillo mínimo para contar como amarillo |
| `area_min_mancha` | 40 | Área mínima de una mancha, en píxeles |
| `umbral_leve`, `umbral_grave` | 5 y 20 | Límites de severidad, en % de área afectada |

## 5. Limitaciones conocidas

- Detecta alteraciones visibles, no diagnostica enfermedades.
- Funciona mejor con fondos neutros o de suelo. Si detrás hay otras hojas, pueden colarse en el análisis.
- Es sensible a la iluminación, así que conviene luz natural homogénea.
