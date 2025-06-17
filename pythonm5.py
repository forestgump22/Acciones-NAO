import pygame
import pygame.camera
from PIL import Image
from pygame.locals import *
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from skimage import measure, filters
from scipy import ndimage
from sklearn.datasets import fetch_openml
from collections import Counter
from naoqi import ALProxy
import vision_definitions

print("Descargando dataset MNIST")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X = X / 255.0
indices_por_clase = {i: [] for i in range(10)}
for idx, label in enumerate(y):
    if len(indices_por_clase[label]) < 500:
        indices_por_clase[label].append(idx)
    if all(len(v) == 500 for v in indices_por_clase.values()):
        break
indices_total = sum(indices_por_clase.values(), [])
X_balanced = X[indices_total]
y_balanced = y[indices_total]
print("Distribucion de clases:", Counter(y_balanced))
x_train, x_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)
print("Entrenando SVM")
model = SVC(C=10, gamma=0.05, kernel='rbf')
model.fit(x_train, y_train)
print("Accuracy del modelo MNIST (28x28): %.4f" % 
      accuracy_score(y_test, model.predict(x_test)))

# ======== PREPROCESADO ========
def procesar_imagen_para_modelo(imagen_pil):
    imagen_pil = imagen_pil.convert("L")
    imagen_pil = ImageEnhance.Brightness(imagen_pil).enhance(1.8)
    imagen_pil = ImageEnhance.Contrast(imagen_pil).enhance(1.5)
    imagen_np = np.array(imagen_pil) / 255.0
    if imagen_np.mean() > 0.5:
        imagen_np = 1.0 - imagen_np
    try:
        thresh = filters.threshold_otsu(imagen_np)
        binaria = imagen_np > thresh
    except ValueError:
        binaria = imagen_np > imagen_np.mean()
    etiquetas = measure.label(binaria)
    regiones = [r for r in measure.regionprops(etiquetas) if r.area > 50]
    if not regiones:
        return np.zeros((1, 28 * 28))
    region = max(regiones, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    PAD = 20
    minr = max(0, minr - PAD)
    minc = max(0, minc - PAD)
    maxr = min(binaria.shape[0], maxr + PAD)
    maxc = min(binaria.shape[1], maxc + PAD)
    recorte = imagen_np[minr:maxr, minc:maxc]
    recorte_pil = Image.fromarray((recorte * 255).astype(np.uint8))
    recorte_pil.thumbnail((20, 20), Image.ANTIALIAS)
    lienzo = Image.new("L", (28, 28), color=0)
    ox = (28 - recorte_pil.width) // 2
    oy = (28 - recorte_pil.height) // 2
    lienzo.paste(recorte_pil, (ox, oy))
    lienzo_np = np.array(lienzo) / 255.0
    cy, cx = ndimage.center_of_mass(lienzo_np)
    sx = int(round(28/2 - cx))
    sy = int(round(28/2 - cy))
    lienzo_np = ndimage.shift(lienzo_np, shift=[sy, sx])
    return lienzo_np.reshape(1, -1)

# ======== LOCAL ========
def usar_imagen_local(ruta, tts):
    try:
        imagen = Image.open(ruta).convert('RGB')
        imagen.show()
        x_in = procesar_imagen_para_modelo(imagen)
        pred = model.predict(x_in)[0]
        texto = "Veo un %d" % pred
        print(texto)
        tts.say(texto)
    except Exception as e:
        print("Error local:", e)


# ======== NAO REAL/SIMULADO ========
def showNaoImage(IP, PORT):
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    for idx in (0,1):
        try:
            vc = camProxy.subscribeCamera("py", idx,
                                          vision_definitions.kVGA,
                                          vision_definitions.kRGBColorSpace,
                                          5)
            naoImage = camProxy.getImageRemote(vc)
            camProxy.releaseImage(vc)
            w, h = naoImage[0], naoImage[1]
            arr = naoImage[6]
            pil = Image.frombytes("RGB", (w, h), arr)
            pil.show()
            x_in = procesar_imagen_para_modelo(pil)
            pred = model.predict(x_in)[0]
            texto = "Veo un %d" % pred
            print(texto)
            tts.say(texto)
            return
        except Exception as ex:
            print("Error camara", idx, ex)
    print("No se obtuvo imagen de NAO")

# ======== MAIN ========
# Todo el codigo funciono con la version de 2.7.16, 
# lo que no se probo como tal es del caso del robot real
# Gracias :D
if __name__ == '__main__':
    # configura IP/PORT de tu NAO (simulado o real)
    NAO_IP   = "127.0.0.1"
    NAO_PORT = 50403
    tts = ALProxy("ALTextToSpeech", NAO_IP, NAO_PORT)

    modo = 'batch'   # 'local', 'webcam', 'nao_real' o 'batch'

    if modo == 'local':
        usar_imagen_local("./imgs/7.jpg", tts)

    elif modo == 'nao_real':
        showNaoImage(NAO_IP, NAO_PORT)

    elif modo == 'batch':
        archivos = ["./imgs/" + str(i) + ".jpg" for i in [1, 2, 5, 6, 7, 8, 9]]
        for s in archivos:
            print("Procesando imagen", s)
            usar_imagen_local(s, tts)

    else:
        print("Modo desconocido. Usa 'local', 'webcam', 'nao_real' o 'batch'.")
