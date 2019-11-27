from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 
class CentroidTracker():
        def __init__(self, maxDisappeared=50):
                # initialize the next unique object ID along with two ordered
                # dictionaries used to keep track of mapping a given object
                # ID to its centroid and number of consecutive frames it has
                # been marked as "disappeared", respectively
                self.nextObjectID = 0
                self.objects = OrderedDict() # 삽입순서를 보존하는 딕셔너리
                self.disappeared = OrderedDict()

                # store the number of maximum consecutive frames a given
                # object is allowed to be marked as "disappeared" until we
                # need to deregister the object from tracking
                self.maxDisappeared = maxDisappeared


        def register(self, centriod):
                self.objects[self.nextObjectID] = centriod
                self.disappeared[self.nextObjectID] = 0
                self.nextObjectID += 1

        def deregister(self, objectID):
                del self.objects[objectID]
                del self.disappeared[objectID]

        def update(self, points):

                if len(points) == 0:
                    # loop over any existing tracked objects and mark them
                    # as disappeared
                   
                    for objectID in list(self.disappeared.keys()):
                        self.disappeared[objectID] += 1
                        #일정 프레임동안 disappeared이면 deregister
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)

                    return self.objects

                
                
                inputCentroids = np.zeros((len(points), 2), dtype = "int")
                

                for(i, (cX,cY)) in enumerate(points):
                        inputCentroids[i] = (cX, cY)
                        

                # 현재 트래킹하고있는 물체가 없다면 추가해줌
                if len(self.objects)== 0 :
                        
                        for i in range(0, len(inputCentroids)):
                                self.register(inputCentroids[i])
                                
                
                else:
                        
                #현재 트래킹하는 물체가 있다면, 현존하는 물체들의 중심과 input centroids의 중심을 매칭시켜줌

                        objectIDs = list(self.objects.keys())
                        objectCentroids = list(self.objects.values())
                        #물체들의 중심과 input centroids간의 거리를 모두  계산
                        # D.shape = (# of object centroids, # of input cen4troids)
                        D = dist.cdist(np.array(objectCentroids), inputCentroids)
                        #1. find the smallest value in each row
                        #2. sort the row indexes based on their min values so that
                        #   the row w/ the smallest value is at the front of the index list
                        # min(axis=1)은 같은 행에 속한 숫자들끼지 비교, argsort는 크기순서를 나타냄
                        # 즉 각 row에 속한 숫자들 중에 가장 작은 숫자를 뽑아내고, 그 숫자들의 크기순서 나타낸 index배열 반환.
                        rows = D.min(axis=1).argsort()
                        #finding the smallest value in each colunm and sorting using row index list
                        # argmin(axis=1)은 같은 행에 속한 숫자들끼지 비교하고 min의 index리턴
                        # index배열
                        cols = D.argmin(axis=1)[rows]

                        #해당 row,col의 인덱스가 이미 사용되었는지 확인
                        usedRows = set()
                        usedCols = set()

                        for (row, col) in zip(rows, cols):
                            if row in usedRows or col in usedCols:#이미 있으면 무시
                                continue

                            objectID = objectIDs[row]
                            self.objects[objectID] = inputCentroids[col] # set its new centroid
                            self.disappeared[objectID] = 0 #reset the disapppeared counter
                            
                            usedRows.add(row) 
                            usedCols.add(col)
                            
                        #difference function : 차집합
                        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                        if D.shape[0] >= D.shape[1]:#objects가 사라졌거나 lost됐다
                            for row in unusedRows:
                                objectID = objectIDs[row]
                                self.disappeared[objectID] += 1
                                
                                if self.disappeared[objectID] > self.maxDisappeared:
                                    self.deregister(objectID)

                        else:#새 object가 생겼다
                            for col in unusedCols:
                                self.register(inputCentroids[col])
                                
                return self.objects
