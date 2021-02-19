import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import math
import os
from skimage.exposure import rescale_intensity
import skimage as skimage

#don't forget to edit path or to paste all the path from root
path="DATASET/RGB_input/"
dataset="blizzard/"

#pre-processing parameters
median_size = 5
gauss_coef = 11
seuil_changement = 10
seuil_ssd = 50
taille_bloc= 16
nbr_images = len(next(os.walk(path+dataset))[2]) ##recuperer le nombre d'images...

#load background image [the first one]
image_fond = cv2.imread(path+dataset+"1.png",0)

#filters application
image_fond = cv2.medianBlur(image_fond, median_size)
image_fond = cv2.GaussianBlur(image_fond,(gauss_coef,gauss_coef),cv2.BORDER_DEFAULT)
w=image_fond.shape[1]
h=image_fond.shape[0]

#init result matrix to store the binary image
res=np.zeros((h,w),dtype=np.uint8)
for i in range(2,int(nbr_images)+1):
    #repeat the pre-process for each image
    image_current = cv2.imread(path+dataset+str(i)+".png",0)
    image_current = cv2.medianBlur(image_current, median_size)
    image_current = cv2.GaussianBlur(image_current,(gauss_coef,gauss_coef),cv2.BORDER_DEFAULT)

    # Apply the image subs
    cv2.absdiff(image_fond,image_current,res)

    #binarisation
    res = np.where(res<seuil_changement,0,255)
    res = res.astype(np.uint8)

    #display result frame
    cv2.imshow("result",res)
    #display the original frame to vizualise
    cv2.imshow("manual",image_current)

    cv2.waitKey(20)
    #decomment this so each the frame i+1 will be compared with the frame i [comparaison with the background image give  (frame 0) shows a better results]
    #image_fond=image_current
cv2.waitKey(0)
