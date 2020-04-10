import cv2 
import numpy as np
# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('C:\Anacondawin\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


def face_extractor(img):
    #Function detect faces and return cropped image
    #If no faces detected it return the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
     
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    
    return cropped_face

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0  

# Collect 100 samples of your face from webcam input
while True:
    
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Save file in specified directory with unique name
        file_name_path = './Swapnil/face' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        
        
        #Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        cv2.imshow("Face Cropper", face)
        
    else:
        print("Face not found")
        pass
    
    if cv2.waitKey(1) == 13 or count == 200:
        break
        
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
