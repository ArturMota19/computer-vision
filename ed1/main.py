import cv2
import numpy as np
import time

# aq so pra mostrar coordenadas de onde fica o olho
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        print(f"Coordenadas: x={x}, y={y}")

def main():
    # lena na escala cinza
    img = cv2.imread("lena.pgm", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("imagem nao encontrada na root")
        return

    # preserva imagem original
    output = img.copy()

    x, y, w, h = 245, 245, 40, 15   # testa -> olho direito
    roi = img[y:y+h, x:x+w]         # regiao de interesse

    # cria janela de animacao e coloca o evento do mouse
    cv2.namedWindow("Animacao")
    cv2.setMouseCallback("Animacao", mouse_event)

    cv2.imshow("Animacao", output)
    cv2.waitKey(1000)  # esperar antes da animacao

    for _ in range(3):
        for dy in range(0, 15, 2):  # mover para baixo
            frame = output.copy()
            frame[y+dy:y+h+dy, x:x+w] = roi
            cv2.imshow("Animacao", frame)
            cv2.waitKey(50)

        for dy in range(15, -2, -2):  # mover para cima (volta)
            frame = output.copy()
            frame[y+dy:y+h+dy, x:x+w] = roi
            cv2.imshow("Animacao", frame)
            cv2.waitKey(50)

    #cv2.waitKey(0)  # fecha no 0
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
