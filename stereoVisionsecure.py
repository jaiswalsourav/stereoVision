import sys
import cv2
import numpy as np
import time
#import imutils
from matplotlib import pyplot as plt
import urllib.request

# Function for stereo vision and depth estimation
#import triangulation as tri
#import calibration

# Mediapipe for face detection
#import mediapipe as mp
import time

#mp_facedetector = mp.solutions.face_detection
#mp_draw = mp.solutions.drawing_utils

# Open both cameras
#cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
#cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)
import time
import math
from adafruit_servokit import ServoKit
#basepin=11
#arm1pin=13
#arm2pin=24
kit=ServoKit(channels=16)
servo=4
L1=25
L2=17

#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(basepin,GPIO.OUT)
#GPIO.setup(arm1pin,GPIO.OUT)
#GPIO.setup(arm2pin,GPIO.OUT)

def arm1_angle(pin):
    #pwm=GPIO.PWM(pin,50)
    #while True:
    kit.servo[pin].angle=90
    #pwm.start(2.5)
    #duty= (angle/18)+2.5
    #GPIO.output(pin,True)
    #pwm.ChangeDutyCycle(duty)
    time.sleep(2)
    
def arm2_angle(pin):
    #pwm=GPIO.PWM(pin,50)
    #while True:
    kit.servo[pin].angle=90
    #pwm.start(2.5)
    #duty= (angle/18)+2.5
    #GPIO.output(pin,True)
    #pwm.ChangeDutyCycle(duty)
    time.sleep(2)
    
def base_angle(pin):
    #pwm=GPIO.PWM(pin,50)
    #while True:
    kit.servo[pin].angle=90
    #pwm.start(2.5)
    #duty= (angle/18)+2.5
    #GPIO.output(pin,True)
    #pwm.ChangeDutyCycle(duty)
    time.sleep(2)

def set_angle(pin,angl):
    #pwm=GPIO.PWM(pin,50)
    #while True:
    kit.servo[pin].angle=angl
    #pwm.start(2.5)
    #duty= (angle/18)+2.5
    #GPIO.output(pin,True)
    #pwm.ChangeDutyCycle(duty)
    time.sleep(2)
    #GPIO.output(pin,False)
    #pwm.ChangeDutyCycle(0)
    
def base(x,z):
    theta=math.atan2(z,x)
    
    return theta

def inverse_kinematics(x,y,z):
    r=math.sqrt(z**2+y**2)
    #theta1=math.atan2(y,x)
    
    #r2=r-L1
    
    #d=math.sqrt(r2**2 + z**2)
    
    beta= math.atan2(y,z)
    
    #alpha =math.acos((L1**2 + L2**2 -d**2) /(2*L1*L2))
    theta2=math.acos((r**2-L1**2-L2**2)/(2*L1*L2))
    alpha=math.atan(L2*math.sin(theta2)/(L2*math.cos(theta2)+L1))
    theta1=beta-alpha                            
    
    return theta1,math.pi-theta2

# Stereo vision setup parameters
frame_rate = 10    #Camera frame rate (maximum at 120 fps)
B = 3.6               #Distance between the cameras [cm]
f = 4.8              #Camera lense's focal length [mm]
alpha = 65        #Camera field of view in the horisontal plane [degrees]
secure=[]
url1='http://192.168.220.216/640x480.jpg'
url2='http://192.168.220.24/640x480.jpg'

classNames = []
classFile = "/home/jaiswal/Desktop/StereoVisionDepthEstimation/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/jaiswal/Desktop/StereoVisionDepthEstimation/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/jaiswal/Desktop/StereoVisionDepthEstimation/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    centre=[]
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    #centre=[]
    xp=0
    yp=0
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            x,y,w,h=box
            centre.append([(x+x+w)/2,(y+y+h)/2])
            #depth=depth_map[(x+x+w)/2,(y+y+h)/2]
            #print("depth"+depth)
            xp=x+x+w/2
            yp=y+y+h/2
            #print(x)
            #print(y)
            #print(w)
            #print(h)
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo,centre,xp,yp


cv_file = cv2.FileStorage()
cv_file.open('/home/jaiswal/Desktop/StereoVisionDepthEstimation/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL


def find_depth(centreR,centreL, frameR, frameL, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frameR.shape
    height_left, width_left, depth_left = frameL.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
        print(f_pixel)
    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = centreR#right_point[0]
    x_left = centreL#left_point[0]

     #CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]
    if disparity==0:
        print("0")
        return None
    # CALCULATE DEPTH z:
    zDepth = (B*f_pixel)/disparity             #Depth in [cm]

    return zDepth

# Main program loop with face detector and depth estimation using stereo vision
#with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

while True:#(cap_right.isOpened() and cap_left.isOpened()):

        #succes_right, frame_right = cap_right.read()
        #succes_left, frame_left = cap_left.read()
    imgResp=urllib.request.urlopen(url1)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img1=cv2.imdecode(imgNp,1)
    imgResp2=urllib.request.urlopen(url2)
    imgNp2=np.array(bytearray(imgResp2.read()),dtype=np.uint8)
    img2=cv2.imdecode(imgNp2,1)

    ################## CALIBRATION #########################################################

    frameR, frameL =undistortRectify(img2, img1)
    #grayR=cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    #grayL=cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    #stereo=cv2.StereoBM_create(numDisparities=16,blockSize=15)
    #disparity=stereo.compute(grayL,grayR)
    #height_right, width_right, depth_right = frameR.shape
    #height_left, width_left, depth_left = frameL.shape

    #if width_right == width_left:
        #f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
    #depth_map=(B*f_pixel)/disparity
    #depth1=depth_map[130,130]
    #print(depth1)
    #cv2.imshow('depth map',depth_map)

    ########################################################################################

        # If cannot catch any frame, break
    if True:

        start = time.time()
            
            # Convert the BGR image to RGB
        #frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        #frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            #results_right = face_detection.process(frame_right)
            #results_left = face_detection.process(frame_left)
        centre_R=[]
        centre_L=[]
        result1, objectInfo1,centre_R,x1,y1 = getObjects(frameR,0.50,0.3,objects=['bottle'])
        #print(objectInfo1)
        #print(x1)
        #secure[0]=x1
        #print(objectInfo1)
        #print(centre_R[0])
        #x=centre_R[0][0]
        #y=centre_R[0][1]
        #depth=depth_map[x,y]
        #print(depth)
        #print(centre_R[8])
        result2, objectInfo2,centre_L,x2,y2 = getObjects(frameL,0.50,0.3,objects=['bottle'])
        #print(objectInfo2)
        #print(x2)
        al=[x1,x2,y1,y2,objectInfo2,objectInfo1]
        #secure[1]=x2
        #secure[2]=y2
        #print(centre_L)
        #classIds1, confs1, bbox1 = net.detect(frame_right,confThreshold=0.45,nmsThreshold=0.2)
        #print(classIds,bbox)
        #centre=[]
        #if len(objects) == 0: objects = classNames
        #objectInfo1 =[]
    #centre=[]
        #if len(classIds1) != 0:
            #for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            #className1 = classNames[classId - 1]
                #x,y,w,h=box
                #centre.append([(x+x+w)/2,(y+y+h)/2])
                #print(centre)
                #print(x)
                #print(y)
                #print(w)
                #print(h)
                #if className1 in objects:
                    #objectInfo1.append([box,className1])
                    #f (draw):
                        #cv2.rectangle(frame_right,box,color=(0,255,0),thickness=2)
                        #cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # Convert the RGB image to BGR
        #frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        #frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


            ################## CALCULATING DEPTH #########################################################

            #center_right = 0
            #center_left = 0

            #if results_right.detections:
                #for id, detection in enumerate(results_right.detections):
                    #mp_draw.draw_detection(frame_right, detection)

                    #bBox = detection.location_data.relative_bounding_box

                    #h, w, c = frame_right.shape

                    #boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    #center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    #cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            #if results_left.detections:
                #for id, detection in enumerate(results_left.detections):
                    #mp_draw.draw_detection(frame_left, detection)

                    #bBox = detection.location_data.relative_bounding_box

                    #h, w, c = frame_left.shape

                    #boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    #center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    #cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)




            # If no ball can be caught in one camera show text "TRACKING LOST"
        if not centre_R or not centre_L:#not (objectInfo1[2]=='bottle') or not (objectInfo2[2]=='bottle'):
                cv2.putText(img2, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(img1, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        else:
            
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                # All formulas used to find depth is in video presentaion
            depth = find_depth(al[0],al[1],frameR,frameL, B, f, alpha)
            print(depth)
            #depth1=depth_map[al[0],al[1]]
            al.append(str(round(depth,1)))
            #if :
                x_target=
                y_target=
                z_target=
                theta=base(x_target,z_target)
                theta1,theta2=inverse_kinematics(x_target,y_target,z_target)
                base_angle(0)
    #arm1_angle(1)
                set_angle(0,math.degrees(theta))
                set_angle(1,math.degrees(theta1))
                set_angle(3,math.degrees(theta2))
                base_angle(0)
                arm1_angle(1)
                arm2_angle(3)    
            #al=[x1,x2,y2,abs(depth)]
            #x=centre_L[0][0]
            #y=centre_L[0][1]
            #depth=depth_map[120,133]
        #cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        #cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            #secure[3]=depth
            #secure.append(al)
            #print("depth"+str(round(depth,1)))
            ##print(secure)
            #print("depth1"+str(round(depth1,1)))



        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        #print("FPS: ", fps)

        #cv2.putText(img2, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        #cv2.putText(img1, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


            # Show the frames
        cv2.imshow("frame right",frameR) 
        cv2.imshow("frame left", frameL)
        #cv2.imshow("frame righ",img2) 
        #cv2.imshow("frame lef", img1)


            # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#for i in secure:
    #print(i,end=' ')
# Release and destroy all windows before termination
#cap_right.release()
#cap_left.release()

cv2.destroyAllWindows()