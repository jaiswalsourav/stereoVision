import sys
import cv2
import numpy as np
import time
#import imutils
from matplotlib import pyplot as plt
import urllib

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
#import mediapipe as mp
import time

#mp_facedetector = mp.solutions.face_detection
#mp_draw = mp.solutions.drawing_utils

# Open both cameras
#cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
#cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B =3.6                #Distance between the cameras [cm]
f = 4.8              #Camera lense's focal length [mm]
alpha = 65        #Camera field of view in the horisontal plane [degrees]

url1='http://192.168.220.216/640x480.jpg'
url2='http://192.168.220.24/640x480.jpg'

classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            x,y,w,h=box
            centre=((x+x+w)/2,(y+y+h)/2)
            print(x)
            print(y)
            print(w)
            print(h)
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo,centre


cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL


def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

    return zDepth

# Main program loop with face detector and depth estimation using stereo vision
#with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

while True:#(cap_right.isOpened() and cap_left.isOpened()):

        #succes_right, frame_right = cap_right.read()
        #succes_left, frame_left = cap_left.read()
    imgResp=urllib.request.urlopen(url1)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,1)
    imgResp2=urllib.request.urlopen(url2)
    imgNp2=np.array(bytearray(imgResp2.read()),dtype=np.uint8)
    img2=cv2.imdecode(imgNp2,1)

    ################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(img2, img)

    ########################################################################################

        # If cannot catch any frame, break
    if True:

        start = time.time()
            
            # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            #results_right = face_detection.process(frame_right)
            #results_left = face_detection.process(frame_left)
        result1, objectInfo1,centre_R = getObjects(frame_right,0.45,0.2)
        result2, objectInfo2,centre_L = getObjects(frame_left,0.45,0.2)
            # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


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
            #if not results_right.detections or not results_left.detections:
                #cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                #cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            #else:
                # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                # All formulas used to find depth is in video presentaion
        depth = tri.find_depth(center_R, center_L, frame_right, frame_left, B, f, alpha)

        cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
        print("Depth: ", str(round(depth,1)))



        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
            #print("FPS: ", fps)

        cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


            # Show the frames
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)


            # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
#cap_right.release()
#cap_left.release()

cv2.destroyAllWindows()