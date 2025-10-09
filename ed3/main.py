import sys
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

def main():
  img_path = 'penguins.png'
  if not os.path.exists(img_path):
    print(f'Imagem não encontrada na root.')
    sys.exit(1)

  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    print('Erro ao carregar a imagem.')
    sys.exit(1)

  Kv = np.array([[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]], dtype=np.float64) # aqui eh o filtro de solbel, pro vertical
  Ku = Kv.T # transposta de Kv, para gradiente horizontal
  # armazenei em kv e ku como pedido pelo professor
  # img original
  Iu = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, Ku)  # gradiente horizontal
  Iv = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, Kv)  # gradiente vertical

  # uint8 pra visualização
  Iu_vis = cv2.convertScaleAbs(Iu)
  Iv_vis = cv2.convertScaleAbs(Iv)

  # pega as bordas usanbdo a formula sqrt(Iu^2 + Iv^2)
  I_mag = np.sqrt(Iu**2 + Iv**2)
  #normalizacao para 0-255 e conversao para uint8
  I_edges = cv2.normalize(I_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

  # salvar as imagens
  cv2.imwrite('Iu.png', Iu_vis)
  cv2.imwrite('Iv.png', Iv_vis)
  cv2.imwrite('edges.png', I_edges)

  # plotando (usei matplotlib pois o cv2.imshow as vezes não funciona direito)
  titles = ['Original', 'Gradiente horizontal (Iu)', 'Gradiente vertical (Iv)', 'Bordas (sqrt(Iu^2+Iv^2))']
  imgs = [img, Iu_vis, Iv_vis, I_edges]
  plt.figure(figsize=(12, 6))
  for i, (t, im) in enumerate(zip(titles, imgs), 1):
    plt.subplot(1, 4, i)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.title(t)
    plt.axis('off')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()