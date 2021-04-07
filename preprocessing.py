import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
import PIL
import PIL.Image
from keras.models import load_model
import h5py
import pathlib
import cv2
import numpy as np
import os
import zipfile



def load_images_from_folder(folder):
    images = []
    i=0
    no_of_faces = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(gray)
            img_with_detections = np.copy(img)
            i+=1
            print(i,"-"," has",len(faces) , " faces in it")
            no_of_faces += len(faces)
            if (len(faces) > 0):
                for (x, y, w, h) in faces:
                    roi_color = img[y:y + h, x:x + w]
                    immg = cv2.resize(roi_color, (150, 150))
                    images.append(immg)
            else: continue


    return images , no_of_faces

folder='F:\\New_folderr\\New folder'

images , no_of_faces= load_images_from_folder(folder)
print(len(images))

print("no_of_faces: ",no_of_faces)

path = "F:\mask detection\preprocessed data/train/with_mask"

for i in range(no_of_faces):
    print(str(i)+"th pic  done ")
    cv2.imwrite(str(path)+str(i)+".jpg" , images[i])
    cv2.waitKey(0)