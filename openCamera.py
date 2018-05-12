# -*- coding: utf-8 -*-
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(cap.isOpened()):
    #BGR image feed from camera
    ret,img=cap.read()
    cv2.imshow('output',img)
    
    #BGR to gray conversion
    
    img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale',img2)
    
    #BGR to binary (RED) thresholded image
    
    imgthreshold=cv2.inRange(img,cv2.Scalar(3,3,125),cv2.Scalar(40,40,255))
    cv2.imshow('thresholded',imgthreshold)
    
    k=cv2.waitKey(10)
    if k==27:
        break


cap.release()
cv2.distroyAllWindows()
    

