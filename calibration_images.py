import cv2
import numpy as np


#cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(2)

num = 0
import urllib.request
url1='http://192.168.220.216/640x480.jpg'
url2='http://192.168.220.24/640x480.jpg'
while True:# cap.isOpened():

    #succes1, img = cap.read()
    #succes2, img2 = cap2.read()
    imgResp=urllib.request.urlopen(url1)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img1=cv2.imdecode(imgNp,1)
    imgResp2=urllib.request.urlopen(url2)
    imgNp2=np.array(bytearray(imgResp2.read()),dtype=np.uint8)
    img2=cv2.imdecode(imgNp2,1)

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img1)
        cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img1)
    cv2.imshow('Img 2',img2)
