import cv2
import numpy as np


video_path = 'traffic_sequence.mpg'

cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
if not ret:
  print("Não foi possível ler o vídeo.")
  cap.release()
  exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
  ret, frame = cap.read()
  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  diff = cv2.absdiff(prev_gray, gray)

  _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

  cv2.imshow('Frame Atual', gray)
  cv2.imshow('Mudancas Detectadas', diff_thresh)
  prev_gray = gray.copy()

  if cv2.waitKey(30) & 0xFF == 27:  # Pressione ESC para sair
    break

cap.release()
cv2.destroyAllWindows()