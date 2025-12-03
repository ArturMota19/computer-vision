import cv2
import numpy as np
import math
import csv

# Constante de entrada e saida
VIDEO_PATH = "video1.mp4"
OUTPUT_CSV = "traj_video1.csv"


# Homografia -> mundo (plano do chão)
H = np.array([[0.9236, 0, 1077.2023],
              [0, -0.9102, 1952.5904],
              [0, 0, 1.0]])

# Matriz intrínseca da câmera -> serve para referência
K = np.array([[2216.537, 0, 1077.202],
              [0, 2184.502, 1952.590],
              [0, 0, 1.0]])

# parâmetros de detecção de cor (HSV) -> nesse caso vermelho
LOWER_RED1 = np.array([0, 70, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170,70,50])
UPPER_RED2 = np.array([180,255,255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

rows = [["frame","time","u","v","X_world","Y_world","yaw_deg","area"]]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t = frame_idx / fps
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        frame_idx += 1
        continue

    # escolhe o maior contorno por área
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:  # filtro de ruído
        frame_idx += 1
        continue

    # retângulo rotacionado
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)
    box = box.astype(np.intp)

    # centro imagem
    M = cv2.moments(c)
    if M["m00"] == 0:
        frame_idx += 1
        continue
    cx = float(M["m10"]/M["m00"])
    cy = float(M["m01"]/M["m00"])

    # calcular yaw a partir de um eixo do retângulo: escolher primeiro aresta do box
    # ordenar box para ter um vetor lateral
    # aqui simplista: usar o lado entre box[0] e box[1]
    p0 = box[0].astype(float)
    p1 = box[1].astype(float)
    dx = p1[0]-p0[0]
    dy = p1[1]-p0[1]
    yaw_img = math.degrees(math.atan2(dy, dx))  # orientação na imagem

    # Mapear centro para o mundo via H^{-1}
    invH = np.linalg.inv(H)
    img_pt = np.array([cx, cy, 1.0])
    world_h = invH @ img_pt
    world_h = world_h / world_h[2]
    Xw, Yw = world_h[0], world_h[1]

    # Alternativa: mapear também um canto para definir orientação no mundo
    img_pt2 = np.array([p0[0], p0[1], 1.0])
    world_h2 = invH @ img_pt2
    world_h2 = world_h2 / world_h2[2]
    Xw2, Yw2 = world_h2[0], world_h2[1]
    yaw_world = math.degrees(math.atan2(Yw2 - Yw, Xw2 - Xw))

    rows.append([frame_idx, t, cx, cy, Xw, Yw, yaw_world, area])

    # desenho para debug
    out = frame.copy()
    cv2.drawContours(out, [box], 0, (0,255,0), 2)
    cv2.circle(out, (int(cx),int(cy)), 4, (255,0,0), -1)
    cv2.putText(out, f"X={Xw:.2f} Y={Yw:.2f} yaw={yaw_world:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # mostrar (tela menor)
    small = cv2.resize(out, None, fx=0.4, fy=0.4)  # redimensiona para 40%
    cv2.imshow("det", small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# salvar CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Pronto! CSV salvo:", OUTPUT_CSV)
