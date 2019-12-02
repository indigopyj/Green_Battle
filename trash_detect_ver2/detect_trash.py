import math
import cv2

class TrashLike:
    
    def __init__(self,yolo_window,blob_point,side):
        self.count=1
        self.car=yolo_window
        self.carSize=yolo_window[1][0]-yolo_window[0][0]
        self.location=blob_point
        self.side=side # left or right
        self.direction=1
        self.trash=False
        self.nearest_blob=[]
        self.age=0
        
    def updateTrashlike(self,location,direction):
        self.age=0
        self.count+=1
        self.location=location
        self.direction=direction  # direction=dy/dx
                                  # if side=='left' this value tends to decrease
                                  # if self=='right' it tends to increase
    def updateNearestBlob(self,blob_points):
        blob_info=[]
        tpoint=self.location
        side=self.side
        for bpoint in blob_points:
            dist=distance(tpoint,bpoint)
            direc=direction(tpoint,bpoint)
            if side=='left' and bpoint[0]<tpoint[0]: # left
                blob_info.append([bpoint,dist,direc])
            elif side=='right' and bpoint[0]>tpoint[0]: # right
                blob_info.append([bpoint,dist,direc])

        blob_info.sort(key=lambda x: x[1]) # 가까운 순으로 정렬
        blob_info=blob_info[0:5] # 가장 가까운 5개
        self.nearest_blob=blob_info

def distance(point1,point2):
    x1,y1=point1
    x2,y2=point2
    dist=math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

def direction(point1,point2):
    x1,y1=point1
    x2,y2=point2
    if x1==x2:
        x1+=0.01
    direc=(y1-y2)/abs(x1-x2)
    return direc

trashlike_list=[]  
                                      
def findTrashlike(yolo_window_list,blob_points):

    for window in yolo_window_list:
        
        left,top=window[0]
        right,bottom=window[1]
            
        width=right-left
        height=bottom-top   
                 
        left_bound=[left-int(width*0.2),left+int(width*0.1)]
        right_bound=[right-int(width*0.1),right+int(width*0.2)]
           
        for bpoint in blob_points:
            
            flag=False
            
            for trashlike in trashlike_list:
                if trashlike.location==bpoint:
                    flag=True # already in trashlike list
                    break

            x,y=bpoint

            if (y<bottom and y>top)and flag==False:
         
                    if x>left_bound[0] and x<left_bound[1]:
                        trashlike_list.append(TrashLike(window,bpoint,'left')) # left
                    if x>right_bound[0] and x<right_bound[1]:
                        trashlike_list.append(TrashLike(window,bpoint,'right')) #right
        
        


def updateTrashlikeList(blob_points):

    for trashlike in trashlike_list:
        
        trashlike.age+=1
        trashlike.updateNearestBlob(blob_points)
        
        max_dist=trashlike.carSize/5
        min_dist=trashlike.carSize/15 # min,max 수치가 적절한지 출력하여 확인해 보기
            
        for blob_point,dist,direct in trashlike.nearest_blob:
            
            if dist<=max_dist and dist>=min_dist:                  
                # 여기서는 기울기 정보를 충분히 이용할 수 없음. -> 기울기 비교하기
                if (trashlike.direction+0.2>direct and trashlike.direction-2<direct ):
                    trashlike.updateTrashlike(blob_point,direct)                        
                    break
                
        if(trashlike.age>3):
            trashlike_list.remove(trashlike)
            
        if(trashlike.count>5):
            trashlike.trash=True
            
            
            
def detectTrash(image):
    
    for trashlike in trashlike_list:
        
        if trashlike.trash==True:
            
            location=trashlike.location
            cv2.putText(image, "trash", location, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(image, location, 15, (0, 255, 0), 2)
            print("trash detected at"+str(location))
            
            
def getTrashlikeList():
    print("trashlike list :",trashlike_list)
    return trashlike_list
            



        

        
        