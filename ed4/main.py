import cv2
import numpy as np
import time


def main():
    # lena na escala cinza
    img = cv2.imread("lena.pgm", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("imagem nao encontrada na root")
        return

if __name__ == "__main__":
    main()
