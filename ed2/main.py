import cv2
import numpy as np

import matplotlib.pyplot as plt

#lendo img
img = cv2.imread('penguins.png', cv2.IMREAD_GRAYSCALE)

# checa path
if img is None:
  raise FileNotFoundError("Confira se a imagem penguins.png está na root, com esse nome.")

# histograma
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# nivel de cinza mais frequente
most_freq_gray = np.argmax(hist)

# plotando
plt.figure(figsize=(10, 4))
plt.title('Histograma da Imagem')
plt.xlabel('Nível de Cinza')
plt.ylabel('Frequência')
plt.plot(hist, color='black')
plt.axvline(most_freq_gray, color='red', linestyle='--', label=f'Mais frequente: {most_freq_gray}')
plt.legend()
plt.show()

# escolha (arbitraria) de um threshold
threshold_value = 120

# valor da imagem binaria logica
_, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

# plotando img binaria logica
plt.figure(figsize=(6, 6))
plt.title('Imagem Binária')
plt.imshow(binary_img, cmap='gray')
plt.axis('off')
plt.show()

# print no console
print(f"Nível de cinza mais frequente: {most_freq_gray}")
print(f"Limite de threshold utilizado: {threshold_value}")