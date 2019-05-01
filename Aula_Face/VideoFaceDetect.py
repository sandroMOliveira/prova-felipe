# -*- coding: UTF-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

def split_video_channels():
    print('''
        Integrantes:
        Daniel Águila,
        Gabrielle Liberato,
        Sandro Oliveira
    ''')
    face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture('coringa.mp4')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    output_face = cv2.VideoWriter('coringa_face_detect.avi', codec, 23.0, size)
    print('Seu vídeo está em processamento, vá tomar um café ou assistir um GOT!')
    while True:
        ret_val, frame = cap.read()
        if frame is None:
            break     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        
        #Para ver o os frames em construção
        # cv2.imshow('Video Face', frame)
        output_face.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('Terminou!')
    cap.release()
    cv2.destroyAllWindows()

split_video_channels()