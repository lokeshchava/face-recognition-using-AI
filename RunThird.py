import cv2
import numpy as np
import webbrowser
import os

face_classifier = cv2.CascadeClassifier('C:\Anacondawin\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

while True:
    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        #print("In Try1")
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        #Pass face to prediction model
        # "results" comprises of a tuple containing the label and the conf
        results = swapnil_model.predict(face)
        print(results)
        
        if results[1] < 500:
            #print("In Try2")
            confidence = int( 100 * (1 - (results[1])/400))
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100,120), font, fontScale, color, thickness, cv2.LINE_AA)
        
        if confidence >= 90:
            #print("In Try3")
            cv2.putText(image, "Hey Swapnil !!", (250, 450), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Face Recognition', image)
        else:
            #print("In Try4")
            cv2.putText(image, "Unkown", (250, 450), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Face Recognition', image)
    except:
        print("In except")
        cv2.putText(image, "No Face Found", (220, 120), font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(image, "Locked", (250, 450),font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Face Recognition', image)
        pass
    
    if cv2.waitKey(1) == 13:
        break
