import numpy as np
import cv2 as cv


#Model
net = cv.dnn.readNetFromCaffe("deploy.prototxt.txt", caffeModel="res10_300x300_ssd_iter_140000.caffemodel") #I am getting an error here


image = cv.imread("C:/Users/hasan/Pictures/Group 3.jpg")
(h, w) = image.shape[:2]
blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0,0,i,2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.putText(image, text, (startX, y), cv.FONT_ITALIC, 0.45, (0, 0, 255), 2)
        
    cv.imshow("Output", image)
    cv.waitKey(0)

