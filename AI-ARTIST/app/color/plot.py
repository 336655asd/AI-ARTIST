#-*- coding=utf-8 -*-
import cv2
import numpy as np
import primary_c
def plot(pic_dir):
    primary_c.temperature(pic_dir)
    img = np.zeros((100,500,3),np.uint8)
    #tem=[(0,0,225),(0,225,0),(225,0,0)]
    with open('app/color/temputre.txt','r') as f:
        i=0
        for t in f.readlines():
            tlist=t.split(',')
            rgb=(int(tlist[0]),int(tlist[1]),int(tlist[2]))
            print rgb
            xx=(i*50,0)
            yy=(i*50+50,100)
            cv2.rectangle(img, xx, yy, rgb, -1)
            i=i+1
    cv2.imwrite("app/static/uploads/tempture.jpg",img)
    #cv2.imshow("img",img)
    #cv2.waitKey(0)
    
    
if __name__ == "__main__":
    plot('feathers.jpg')
