import cv2
import matplotlib.pyplot as plt

# le img
img = cv2.imread('flowers9.png')

# open cv pega BRG, converter para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# separa os planos
r, g, b = cv2.split(img_rgb)

# exibit a img completa e cada canal separad
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.imshow(img_rgb)
plt.title('Imagem RGB')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(r, cmap='Reds')
plt.title('Canal R (Vermelho)')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(g, cmap='Greens')
plt.title('Canal G (Verde)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(b, cmap='Blues')
plt.title('Canal B (Azul)')
plt.axis('off')

# histograma p cada canal
plt.subplot(2, 1, 2)
plt.hist(r.ravel(), bins=256, color='r', alpha=0.5, label='Vermelho')
plt.hist(g.ravel(), bins=256, color='g', alpha=0.5, label='Verde')
plt.hist(b.ravel(), bins=256, color='b', alpha=0.5, label='Azul')
plt.title('Histogramas dos Canais de Cor')
plt.xlabel('Intensidade')
plt.ylabel('Número de Pixels')
plt.legend()

plt.tight_layout()
plt.show()
