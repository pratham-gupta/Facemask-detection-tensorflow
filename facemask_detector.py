from imutils.video import VideoStream
import cv2
import imutils
import time
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
prototxtpath = './face-mask-detector/face_detector/deploy.prototxt'
weightspath = './face-mask-detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
facenet = cv2.dnn.readNet(prototxtpath,weightspath)
color = (0, 255, 0)
masknet = load_model('./model2.h5')
if __name__ == '__main__':

    vs = VideoStream(0).start()
    time.sleep(1)


    while(True):

        frame = vs.read()
        frame = imutils.resize(frame,width=800)


        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
        facenet.setInput(blob)
        detections = facenet.forward()
        faces = []
        locs = []
        preds = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > 0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY) = box.astype('int')
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) >0:
            faces = tf.convert_to_tensor(faces,dtype='float32')
            preds = masknet.predict(faces,batch_size=32)

        for box,pred in zip(locs,preds):
            startX,startY,endX,endY = box
            (Mask,withoutMask) =  pred
            label = 'Mask' if Mask>withoutMask else "No Mask"
            color = (0,255,0) if label == 'Mask' else (0,0,255)
            label = "{}: {:.2f}%".format(label, max(Mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)





            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame",frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    cv2.destroyAllWindows()
    vs.stop()
