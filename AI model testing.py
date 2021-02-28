# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:10:10 2020

@author: akshada
"""

from keras.models import load_model
import numpy as np
import cv2
import pandas as pd
data = pd.read_csv("data_asl.csv")
model = load_model('asl_pred_50.h5')

from skimage.transform import resize
def detect(frame):
    img = resize(frame, (64,64,1))
    img = np.expand_dims(img, axis=0)
    if(np.max(img)>1):
        img=img/255.0
    prediction = model.predict(img)
    #print(prediction)
    prediction = model.predict_classes(img)
    print(prediction)
    for i in range(len(data)):
        if (prediction == data['Alp'][i]):
            d = data['Index'][i]
            print(d)
            a = ""
            a=a+d
    
f1 = cv2.imread(r"C:\Users\akshada\Downloads\N_test.jpg")
detect(f1)


pred = detect(cv2.imread(r'C:\Users\akshada\Downloads\dataset-20201102T212714Z-001\dataset\test'))
print()

word = ""
import cv2
a = cv2.VideoCapture(0)
while (a.isOpened()):
    r, f = a.read()
    
    if (r==True):
        detect(f)
   
    cv2.imshow("Capturing", f)
    key=cv2.waitKey(10)
        
    if key == ord('q'):
        cv2.destroyAllWindows()
        a.release()
        #out.release()