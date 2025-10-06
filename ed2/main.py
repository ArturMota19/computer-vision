import cv2
import numpy as np

import matplotlib.pyplot as plt

#lendo img
img = cv2.imread('penguins.png', cv2.IMREAD_GRAYSCALE)

# checa path
if img is None:
  raise FileNotFoundError("Confira se a imagem penguins.png está na root, com esse nome.")

# Obtenha o histograma da imagem
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Determine o nível de cinza mais frequente
most_freq_gray = np.argmax(hist)

print(f"Nível de cinza mais frequente: {most_freq_gray}")
