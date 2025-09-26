#modified the data collection slightly to be its frontend temporarily until i created something better

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0) # Creates a video feed from your webcam, 0 means default, 1-3 or some numbers if you hvae multiple webcams
detector = HandDetector(maxHands=1) # Hand Detector class
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

folder = "Data/143"
counter = 0

labels = ["A", "B", "C", "i <3 u"]

while True:
    sucess, img = cap.read() # Capture a frame from the video feed, returns 2 values | success: a boolean (True/False) indicating if it read the frame correctly. img: the actual image/frame captured.
    hands, img = detector.findHands(img) # Detect hands and draws landmark around it

    if hands: #if there is a hand
        hand = hands[0] # the hand (0  because it is a list of hands and we have 1 hand)
        x, y, w, h = hand['bbox'] # gets the size of the bounding box
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] # crops the hand

        # if imgCrop.size > 0: #sometimes the crop image is empty, prevents the error when you remove ur hand on the screen resulting in an empty crop image
        #     cv2.imshow("Crop Image", imgCrop) #shows the crop image in a window 
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #creates a image with (size), (dataType) parameters in tuples, np.uint is rgb

        imgCropShape = imgCrop.shape
        
        
        aspectRatio = h/w #something to do with centering the crop img within the white img, i still don't get it
        if aspectRatio > 1: 
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape #up to this code, it makes sure that the img height is always 300, from what it understand

            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+ wGap] = imgResize #overlays the crop img on top of the white img + makes sure its always centered
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction)

            cv2.putText(img, labels[index], (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 3)

            
        else:

            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape 

            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+ hGap, :] = imgResize #makes sure the img width is always 300

        cv2.imshow("White Image", imgWhite) #shows it again

    cv2.imshow("Bomboclat", img) # Displays the current image in a window title Bomboclat
    key = cv2.waitKey(1) # Waits for a key pressed for 1 millisecond, if not, the program continues
    if key == ord("q"): # stops the program if you pressed q
        break

