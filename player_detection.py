import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy import argmax


model = load_model(r'model4.h5')
net = cv2.dnn.readNet('files/yolov3.weights','files/yolov3.cfg')
classes=[]
with open('files/coco.names','r') as f:
    classes= [line.strip() for line in f.readlines()]
layer_names= net.getLayerNames()
output_layers= [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

ground= cv2.imread('files/dst.jpg',-1)
cap= cv2.VideoCapture('files/video.avi')
ground= ground[0:600, 300:600]
ground= cv2.resize(ground, (900, 600))
while True:

    #img = cv2.imread('room_ser.jpg')
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    ret, img= cap.read()
    img= cv2.resize(img, (900, 600))
    ground_copy= ground.copy()
    height,width,_ = img.shape

    blob= cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs= net.forward(output_layers)
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores= detection[5:]
            class_id= np.argmax(scores)
            confidence= scores[class_id]
            if confidence > 0.5:
                center_x= int(detection[0]* width)
                center_y= int(detection[1]* height)
                w= int(detection[2]* width)
                h= int(detection[3]* height)

                x= int(center_x - w/2)
                y= int(center_y - h/2)

                #cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    number_objects_detected= len(boxes)
    indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label= str(classes[class_ids[i]])
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(img,label,(x,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (96, 96))
            ym = model.predict(np.reshape(roi, (1, 96, 96, 3)))
            ym = argmax(ym)

            x1 = w / 2
            y1 = h / 2
            cx = x + x1
            cy = y + y1

            if ym == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(ground_copy, (int(cx), int(cy)), 16, (255, 0, 0), -1)
                cv2.putText(img, 'team_1', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            elif ym == 1:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(ground_copy, (int(cx), int(cy)), 16, (0, 255, 0), -1)
                cv2.putText(img, 'team_2', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            elif ym == 2:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(ground_copy, (int(cx), int(cy)), 16, (0, 0, 255), -1)
                cv2.putText(img, 'referee', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



            #cv2.circle(ground_copy, (int(cx), int(cy)), 16, (0, 255, 0), -1)


    cv2.imshow('Image',img)
    cv2.imshow('2DlayOut', ground_copy)
    #print(img.shape)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
