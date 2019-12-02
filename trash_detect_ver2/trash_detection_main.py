from yolo_pipeline2_tiny import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from detect_trash import *

PI=math.pi

def pipeline_yolo(img):   
    output, window_list = vehicle_detection_yolo(img)
    return output, window_list
   

def filter_mask(img): # noise removal & dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # Fill any small holes
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) # Remove noise
    dilation = cv2.dilate(opening, kernel, iterations=2) # Dilate to merge adjacent blobs

    return dilation

def blobDetection(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)    
    
    params = cv2.SimpleBlobDetector_Params()
    
    
    params.minThreshold = 50;
    params.maxThreshold = 255;

    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 1000
    #Circularity filter
    params.filterByCircularity = False
    #Convexity filter
    params.filterByConvexity = False
    #Inertia filter
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(threshold)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, \
                                          np.array([]), (0,0,255),\
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    keypoints2=[] # 블롭 좌표
    for keypoint in keypoints:
        x = round(keypoint.pt[0])
        y = round(keypoint.pt[1])
        keypoints2.append((x,y))

    return keypoints2, im_with_keypoints


if  __name__ == "__main__":

    #COUNTER = 0
    #CONSEC_FRAMES = 8
    #ALARM_ON = False
    
    
    cap = cv2.VideoCapture("input_video/project_video_2.mp4")
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter('detection_output.mp4', fourcc, 25.0, (640,480))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100, detectShadows=True)
    
    count=0
    
    #f=open('output_video/project_video_2_coord.txt','w')

    while (cap.isOpened()):
        count+=1
        
        ret,frame = cap.read()
        if ret is False:
            break
        
        result, window_list = pipeline_yolo(frame) 
        print("yolo window list at frame"+str(count)+" : "+str(window_list))
        
        
        
        mask = bg_subtractor.apply(frame) # backround subtraction
        mask = filter_mask(mask) # noise removal     
        # keypoints = [] # list of centroid coordinates of blobs
        keypoints, mask = blobDetection(mask) # blob detection
        print("blob points at frame"+str(count)+" : "+str(keypoints))
        
        
        
        updateTrashlikeList(keypoints)
        findTrashlike(window_list,keypoints)
        
        detectTrash(mask)
        
        for window in window_list:
            cv2.rectangle(mask,window[0],window[1],(255,0,0),10)
       
        
        #out.write(frame)
        cv2.imshow("result", mask)
        cv2.imwrite("blob_detection/frame"+str(count)+".png",mask)

        
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    #cv2.destoryAllWindows()

    
