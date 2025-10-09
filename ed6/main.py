import cv2
import numpy as np

import matplotlib.pyplot as plt

# img1 - imagem em tons de cinza e formato double
img = cv2.imread('flowers4.png', cv2.IMREAD_GRAYSCALE)
img_double = img.astype(np.float64) / 255.0

plt.figure(figsize=(6, 6))
plt.title('Imagem em tons de cinza (double)')
plt.imshow(img_double, cmap='gray')
plt.axis('off')
plt.show()

# img2 - kernel 15x15 (média)
kU = np.ones((15, 15), dtype=np.float64) / (15 * 15)

plt.figure(figsize=(4, 4))
plt.title('Kernel Médio 15x15')
plt.imshow(kU, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()

# img3- aplicar o kernel médio à imagem
imU = cv2.filter2D(img_double, -1, kU)

# img4 - kernel gaussiano 17x17 (sigma=5)
# sigma eh o desvio padrão da gaussiana
# half_width é a metade do tamanho do kernel (tamanho = 2*half_width + 1)
# serve pra definir o tamanho do kernel, quanto maior o half_width, mais suave a imagem
def kgauss(sigma, half_width):
  size = 2 * half_width + 1
  ax = np.arange(-half_width, half_width + 1)
  xx, yy = np.meshgrid(ax, ax)
  kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
  kernel /= np.sum(kernel)
  return kernel

kG = kgauss(sigma=5, half_width=8)

plt.figure(figsize=(4, 4))
plt.title('Kernel Gaussiano 17x17 (sigma=5)')
plt.imshow(kG, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()

# img5 - aplicar o kernel gaussiano à imagem
imG = cv2.filter2D(img_double, -1, kG)

# img6 - imgs resultantes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Imagem com Kernel Médio (imU)')
plt.imshow(imU, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagem com Kernel Gaussiano (imG)')
plt.imshow(imG, cmap='gray')
plt.axis('off')

plt.show()