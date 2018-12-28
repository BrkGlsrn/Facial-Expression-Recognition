# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:49:29 2018

@author: TCBGULSEREN
"""
import facial_expression_recognition
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor = 1.5,
                                          minNeighbors = 5)
    for (x,y,w,h) in faces :
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        img_item = "my_image.png"
        cv2.imwrite(img_item,roi_gray)
        cv2.rectangle(img , (x,y), (x+w,y+h),(255,0,0),2)
        #cv2.line(img,(x,y),(x+w,y+h),(255,255,255),15)
        indis = facial_expression_recognition.emotion_graph(img_item)
        durum = objects[indis]
        cv2.putText(img, durum, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)                    
        
    cv2.imshow('Face',img)
    if cv2.waitKey(20) & 0xff ==ord('q'):  
        break
    
cap.release()
cv2.destroyAllWindows()