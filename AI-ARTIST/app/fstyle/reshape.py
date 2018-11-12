# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import os 
def resize(filedir):
    file_name=os.getcwd()+"/app/static/uploads/"+filedir
    print("this file is from :{}".format(file_name))
    img=cv2.imread(file_name)
    height=img.shape[0]
    length=img.shape[1]
    print height
    print("**************")
    if height*length>500*500:
        m=max(length,height)
        rate = 500/m
        img2=cv2.resize(img,(int(length*rate),int(height*rate)))
        cv2.imwrite(file_name,img2)
    return file_name

def style_size():
        
        
    return 


