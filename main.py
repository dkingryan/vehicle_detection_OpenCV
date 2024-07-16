import cv2
import numpy as np

video = ("video.mp4")
cap=cv2.VideoCapture(video)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=400)

while True:
    ret,frames = cap.read()
    height, width, _ = frames.shape
    roi = frames[200:1000, 150:1300]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 55:
            #cv2.drawContours(roi, [cnt], -1, (255,0,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w+1,y+h+1), (255,200,0), 3)

    cv2.imshow('Frame', frames)
    #cv2.imshow('ROI', roi)
    #cv2.imshow('Mask', mask)

    if cv2.waitKey(33) == 27:
        break


cv2.destroyAllWindows()

