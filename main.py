import cv2
import numpy as np
import face_recognition


imgsai=face_recognition.load_image_file('ImageBasic/sainath.jpg')
imggopi=face_recognition.load_image_file('ImageBasic/Bhargav.jpg')


imgsai=cv2.cvtColor(imgsai,cv2.COLOR_BGR2RGB)
imggopi=cv2.cvtColor(imggopi,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgsai)[0]
encodesai=face_recognition.face_encodings(imgsai)[0]
cv2.rectangle(imgsai,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLoc1=face_recognition.face_locations(imggopi)[0]
encodegopi=face_recognition.face_encodings(imggopi)[0]
cv2.rectangle(imggopi,(faceLoc1[3],faceLoc1[0],faceLoc1[1],faceLoc1[2]),(255,0,255),2)


results=face_recognition.compare_faces([encodesai],encodegopi,)
faceDis=face_recognition.face_distance([encodesai],encodegopi,)
print(results,faceDis)
cv2.putText(imggopi,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))



cv2.imshow('sainath',imgsai)
cv2.imshow('bhargav',imggopi)

cv2.waitKey(0)
