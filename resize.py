import cv2
import numpy as np

path = "videos/video.mp4"

cap = cv2.VideoCapture(path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 5, (640,480))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
