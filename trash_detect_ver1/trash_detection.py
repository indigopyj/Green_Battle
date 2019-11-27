from yolo_pipeline2_tiny import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tracking import *
import math 

def pipeline_yolo(img):
    
    
    output, window_list, centroid_list = vehicle_detection_yolo(img)

    return output, window_list, centroid_list
    
def filter_mask(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # Fill any small holes
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) # Remove noise

    dilation = cv2.dilate(opening, kernel, iterations=3) # Dilate to merge adjacent blobs

    

    return dilation

def distance(point1,point2):
    x1,y1=point1
    x2,y2=point2
    dist=math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

def blobDetection(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 50;
    params.maxThreshold = 255;

    params.filterByArea = True
    params.minArea = 40
    params.maxArea = 600
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

    return keypoints, im_with_keypoints
    

if  __name__ == "__main__":
        ct = CentroidTracker()
        (H, W) = (None, None)

        COUNTER = 0
        #CONSEC_FRAMES = 8
        #ALARM_ON = False
        TRASH_OUT = False
        FRAME_INDEX = 0
    
        cap = cv2.VideoCapture("examples/blackbox/trash (1).mp4")
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=100, detectShadows=True)
        

        ids_in_box_Past = []
        ids_in_box_Now = []

        trash_id = 0
        
        while (cap.isOpened()):
            _,frame = cap.read()

    
            if W is None or H is None:
                    (H,W) = frame.shape[:2]

            result, window_list, centroid_list = pipeline_yolo(frame) # window_list returns ((x-w, y-h), (x+w, y+h)) 양쪽모서리
            mask = bg_subtractor.apply(result)
            mask = filter_mask(mask) #remove noise

            keypoints = [] # list of centroid coordinates of blobs
            keypoints, mask = blobDetection(mask)

            

            pts = np.asarray([[p.pt[0], p.pt[1]] for p in keypoints])
            

            objects = ct.update(pts)
            
            
            
            for(objectID, centroid) in objects.items():#tracking keypoints
                text = "ID {}".format(objectID)
                cv2.putText(mask, text, (centroid[0] - 10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                cv2.circle(mask, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            ids_in_box_Now=[]
            
            
            for bp in window_list:#classfiy which keypoints are in which box
                cv2.rectangle(mask,bp[0],bp[1],(255,0,0),10)
                
                width = bp[1][0] - bp[0][0]
                for (objectID, centroid) in objects.items():
                    if centroid[0] > bp[0][0]*0.8 and centroid[0] < bp[1][0]*1.2and centroid[1] > bp[0][1] and centroid[1] < bp[1][1]:
                        ids_in_box_Now.append(centroid[1])
                
            
            trash = []
            if ids_in_box_Now and ids_in_box_Past and (len(ids_in_box_Now) == len(ids_in_box_Past)):
                Now = np.asarray(ids_in_box_Now)
                Past = np.asarray(ids_in_box_Past)
                print("Now : "+str(Now))
                print("Past : "+str(Past))
                print()
                
                diff_array = Now - Past
                if diff_array.size > 0 :
                    for diff in diff_array:
                        if diff >= 15:
                            trash.append(diff) 
                            print("*********trash*********")
                            COUNTER += 1
            if COUNTER >= 4:
                COUNTER = 0
                TRASH_OUT = False
                cv2.putText(mask, "TRASH THROWING", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
                print("***********Trash throwing**************")     


            ids_in_box_Past = ids_in_box_Now

            

            cv2.imshow("result", mask)
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break

        cap.release()
