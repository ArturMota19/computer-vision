import cv2
import numpy as np

def brightness(img, value):
  # funçao pra brilho, soma um valor a todos os pixels
  return cv2.add(img, value)

def contrast(img, factor):
  # funçao pra contraste, multiplica todos os pixels por um fator
  return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def negative(img):
  # funçao pra negativo, inverte os valores dos pixels
  return 255 - img

def posterisation(img, levels):
  # funçao pra posterizaçao, reduz o numero de niveis de cinza
  step = 256 // levels
  return (img // step) * step

def main():
  img_uint8 = cv2.imread("lena.pgm", cv2.IMREAD_GRAYSCALE)
  if img_uint8 is None:
    print("imagem nao encontrada na root")
    return
  
  img_double = img_uint8.astype(np.float64) / 255.0
  # todos os valores foram testados e ajustados para melhor visualizaçao (arbitrario)
  bright_uint8 = brightness(img_uint8, 50)
  contrast_uint8 = contrast(img_uint8, 1.5)
  negative_uint8 = negative(img_uint8)
  poster_uint8 = posterisation(img_uint8, 8)

  bright_double = np.clip(img_double + 0.2, 0, 1)
  contrast_double = np.clip((img_double - 0.5) * 1.5 + 0.5, 0, 1)
  negative_double = 1.0 - img_double
  poster_double = np.floor(img_double * 8) / 8

  cv2.imwrite("bright_uint8.png", bright_uint8)
  cv2.imwrite("contrast_uint8.png", contrast_uint8)
  cv2.imwrite("negative_uint8.png", negative_uint8)
  cv2.imwrite("poster_uint8.png", poster_uint8)

  cv2.imwrite("bright_double.png", (bright_double * 255).astype(np.uint8))
  cv2.imwrite("contrast_double.png", (contrast_double * 255).astype(np.uint8))
  cv2.imwrite("negative_double.png", (negative_double * 255).astype(np.uint8))
  cv2.imwrite("poster_double.png", (poster_double * 255).astype(np.uint8))

if __name__ == "__main__":
  main()
