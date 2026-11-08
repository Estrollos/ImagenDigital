import cv2
import numpy as np

# ---------------- PARÁMETROS  ----------------
separar = 7               # erosión para separar la hoja de objetos pegados
borde = 7                 # píxeles del borde de la hoja que no se analizan
h_marron_max = 20         # por debajo de este matiz = marrón
corte_fijo = 30           # corte verde/amarillo mínimo (hojas casi todo amarillas)
margen_verde = 6          # cuánto por debajo del verde dominante empieza lo amarillento
v_min_color = 110         # por debajo de este brillo no se considera amarillo (sombras)
area_min_mancha = 40      # píxeles: descarta ruido pequeño
tam_blur = 5              # suavizado
limpieza = 3              # tamaño del kernel de limpieza de ruido (erosión + dilatación)
umbral_leve, umbral_grave = 5, 20   # % de área afectada

fuente = cv2.FONT_HERSHEY_SIMPLEX
contorno_amarillo = (0, 255, 255)
contorno_marron = (0, 0, 255) 
contorno_hoja = (255, 200, 0)
grosor = 2
ANCHO = 700


# ---------------- PREPROCESADO ----------------
def escalar(img):
    alto, ancho = img.shape[:2]
    if ancho != ANCHO:
        metodo = cv2.INTER_AREA if ancho > ANCHO else cv2.INTER_CUBIC
        img = cv2.resize(img, (ANCHO, int(alto * ANCHO / ancho)), interpolation=metodo)
    k = max(1, tam_blur)
    return cv2.blur(img, (k, k))


def mayor_contorno_relleno(mascara): # se queda solo con el objeto más grande y rellena sus huecos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    salida = np.zeros_like(mascara)
    if contornos:
        cv2.drawContours(salida, [max(contornos, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return salida


# ---------------- SEPARACION HOJA / FONDO ----------------
def separacion_hoja(hsv):
    h, s, v = cv2.split(hsv)
    _, mascara = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((max(1, separar), max(1, separar)), np.uint8)
    mascara = cv2.erode(mascara, k)            # separa objetos pegados y quita tallos finos
    hoja = mayor_contorno_relleno(mascara)     # objeto más grande, sin huecos
    return cv2.dilate(hoja, k)                 # recupera el tamaño


# ---------------- CLASIFICACIÓN POR COLOR ----------------
def corte_verde_amarillo(h, candidatos): # separa verde de amarillo en la hoja
    hist = cv2.calcHist([h], [0], candidatos, [180], [0, 180])
    pico = int(np.argmax(hist))
    return max(corte_fijo, pico - margen_verde)


def clasificar(hsv, hoja):
    h, s, v = cv2.split(hsv)
    k = max(1, borde)
    interior = cv2.erode(hoja, np.ones((k, k), np.uint8))   # quita el borde mezclado con el fondo

    brillante = cv2.inRange(v, v_min_color, 255)
    candidatos = cv2.bitwise_and(cv2.bitwise_and(interior, brillante),
                                 cv2.inRange(h, h_marron_max, 179))
    corte = corte_verde_amarillo(h, candidatos)

    marron = cv2.bitwise_and(interior, cv2.inRange(h, 0, h_marron_max - 1))
    amarillo = cv2.bitwise_and(candidatos, cv2.inRange(h, h_marron_max, max(h_marron_max, corte)))
    verde = cv2.bitwise_and(interior, cv2.bitwise_not(cv2.bitwise_or(amarillo, marron)))

    k2 = np.ones((max(1, limpieza), max(1, limpieza)), np.uint8)   # limpieza de ruido
    amarillo = cv2.dilate(cv2.erode(amarillo, k2), k2)
    marron = cv2.dilate(cv2.erode(marron, k2), k2)
    return interior, verde, amarillo, marron, corte


# ---------------- MANCHAS ----------------
def severidad(porcentaje):
    if porcentaje < umbral_leve:
        return "LEVE"
    if porcentaje < umbral_grave:
        return "MODERADA"
    return "GRAVE"


def dibujar_manchas(img, mascara, color):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contador = 0
    for contorno in contornos:
        if cv2.contourArea(contorno) >= area_min_mancha:
            contador += 1
            cv2.drawContours(img, [contorno], -1, color, grosor)
    return contador


def analizar(img_original):
    img = escalar(img_original)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hoja = separacion_hoja(hsv)
    interior, verde, amarillo, marron, corte = clasificar(hsv, hoja)

    area_hoja = max(cv2.countNonZero(interior), 1)
    pct_verde = 100 * cv2.countNonZero(verde) / area_hoja
    pct_amarillo = 100 * cv2.countNonZero(amarillo) / area_hoja
    pct_marron = 100 * cv2.countNonZero(marron) / area_hoja
    pct_total = pct_amarillo + pct_marron

    resultado = img.copy()
    contornos_hoja, _ = cv2.findContours(hoja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultado, contornos_hoja, -1, contorno_hoja, 1)
    n_am = dibujar_manchas(resultado, amarillo, contorno_amarillo)
    n_ma = dibujar_manchas(resultado, marron, contorno_marron)

    texto = f"Afectada: {pct_total:.1f}%  Severidad: {severidad(pct_total)}"
    texto2 = f"Zonas amarillas: {n_am}  Manchas marrones: {n_ma}"
    cv2.putText(resultado, texto, (10, 28), fuente, 0.7, (0, 0, 0), 4)
    cv2.putText(resultado, texto2, (10, 56), fuente, 0.6, (0, 0, 0), 4)

    datos = dict(verde=round(pct_verde, 2), amarillo=round(pct_amarillo, 2), marron=round(pct_marron, 2), afectada=round(pct_total, 2), zonas_amarillas=n_am, manchas_marrones=n_ma, corte_matiz=corte, severidad=severidad(pct_total))
    return resultado, datos, hoja, amarillo, marron


# ---------------- BARRAS DE AJUSTE ----------------
def poner_barras(ventana):
    def fijar(nombre):
        def f(valor):
            globals()[nombre] = valor
        return f
    barras = [("Separar objetos", "separar", 40), ("Borde", "borde", 30), ("Area min mancha", "area_min_mancha", 500),  ("Blur", "tam_blur", 25), ("Limpieza", "limpieza", 15)]
    for etiqueta, variable, maximo in barras:
        cv2.createTrackbar(etiqueta, ventana, globals()[variable], maximo, fijar(variable))


if __name__ == "__main__":
    img = cv2.imread('./hoja6.jpg')
    if img is None:
        print("No se puede leer la imagen: revisa la ruta")
        exit()
    ventana = "Resultado"
    cv2.namedWindow(ventana)
    poner_barras(ventana)
    while True:
        resultado, datos, hoja, amarillo, marron = analizar(img)
        cv2.imshow('Original', escalar(img))
        cv2.imshow(ventana, resultado)
        cv2.imshow('Mascara amarillo', amarillo)
        cv2.imshow('Mascara marron', marron)
        tecla = cv2.waitKey(50)
        if tecla == ord('q'):
            break
    cv2.destroyAllWindows()