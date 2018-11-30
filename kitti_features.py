from __future__ import division
import cv2
import matplotlib.pyplot as plt
from random import shuffle
from kitti_utils import KittiAnnotation
import numpy as np
import csv
import pickle
import random
import motmetrics as mm
from math import*

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)     
def getArea(A):
    return abs(A[2] - A[0]) * abs(A[3] - A[1])
def IntersectionOverUnion(box1,box2,boxInt):

    inter = getArea(boxInt)
    
    union = getArea(box1) + getArea(box2) - inter
    print ('a',getArea(box1) ,'b',getArea(box2))
    print ("gesuuuuuuuuuuu",inter, 'union', union )
    return inter/union

def findIntersection(A,B):
    C = np.zeros((height,width))
    areaOfIntersection = 0
    for i in range(height):
        for j in range(width):
            if (A[i,j] == 1) and (B[i,j] == 1):
                areaOfIntersection = areaOfIntersection +1
                C[i,j] = 1
    return areaOfIntersection

def findUnion(A,B):
    D = np.zeros((height,width))
    areaOfUnion = 0 
    for i in range(height):
        for j in range(width):
            if (A[i,j] == 1) or (B[i,j] == 1):
                areaOfUnion = areaOfUnion + 1
                D[i,j] = 1
    return areaOfUnion

def getIntersectionOverUnion(A,B):
    IoU = findIntersection(A,B)/findUnion(A,B)
    return IoU

def getBestIou(G,Arrays):
    selectedBox = None
    bestIou = 0
    for k in range(len(Arrays)):
            B = Arrays[k]['img']
            Iou = getIntersectionOverUnion(G,B)
            if Iou > bestIou:
                bestIou = Iou
                selectedBox = k
                centroid = Arrays[k]['centroid']
    return selectedBox,bestIou,centroid


def GetBestIouMask(G,M):
    selectMask = None
    bestIou = 0
    for k in range(len(M)):
        B = M[k]['img']
        Iou = getIntersectionOverUnion(G,B)
        if Iou > bestIou:
                bestIou = Iou
                selectMask = k

    return selectMask,bestIou
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def confronto(A,B):
    currentmask = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            currentmask[i,j] = bb_intersection_over_union(A[i],B[j])
    return currentmask
def confrontoFeatures(A,B):
    currentmask = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            currentmask[i,j] = cosine_similarity(A[i],B[j])
            print ("AOOOOOOOO",currentmask[i,j])
    return currentmask

def getMaximumValue(A):
    maxV = -1
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > maxV:
                maxV = A[i][j]
                row = i
                col = j
    
    eraseColumnAndRow(A,row,col) 
    return maxV,row,col

def eraseColumnAndRow(A,r,c):
    for i in range(len(A)):#colonna
        A[i][c] = 0
    for j in range(len(A[0])):#riga
        A[r][j] = 0
    
 

def findTracksF(vector,listOfTracksF):

    found = -1 
    for k in range(len(listOfTracksF)):
        if np.array_equal(vector,listOfTracksF[k].boxes[len(listOfTracksF[k].boxes) - 1]):
            found = k
    return found 



class Track:
  def __init__(self,id = None,frame = None):

    self.id = id
    self.boxes = []
    self.sequence = []
    self.color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
    self.countNoMatch = 0
    self.state = 'active'
    self.frame = frame 
  def delete(self):
    self.countNoMatch = self.countNoMatch + 1 
    if self.countNoMatch > 2:
        print (' traccia',self.id,'morta')
        self.state = 'death'

def square_rooted(x):

    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
 
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def get_colormap(N):
    cm = plt.cm.get_cmap(None, N)
    colors = [cm(x) for x in range(N)]
    for i in range(N):
        colors[i] = [x * 255 for x in colors[i][:3]]
    # randomize colormap
    shuffle(colors)
    return colors


def main():

    dataset_path = '/Users/andreasimioni/Desktop/kitti/kitti5'
    video_id = '0004'
    file_path = dataset_path + '/label_02/' + video_id + '.txt'
    img_path = dataset_path + '/image_02/' + video_id
    annot = KittiAnnotation(file_path, img_path)
    


    '''
    "gen" is a generator to obtain relevant data about each frame in the sequence 
    use loop=True to loop indefinitely over the video
    you can pass to data_from_generator the types you want the generator to yield
    available types:
    'img' -> the BGR frame
    'annot' -> the ground truth data
    'dets' -> Mask-RCNN detections
    'masks' -> segmentation masks from Mask-RCNN
    '''
    data_from_generator = ('img', 'annot', 'dets','masks','feats')
    gen = annot.annot_generator(data=data_from_generator, loop=False)

    FPS = 30
    colors_tracks = get_colormap(annot.num_tracks)
    colors_dets = get_colormap(5000)
    nCiclo = 0
    PredictionedBoxes = []
    groundTruthBoxes = []
    predictionedMasks = []
    pastboxes = []
    maybeBoxes = []
    currentID = 0
    featureVectors = []
    pastVector = []
    listTracks = []
    listTracksF = []
    groundTracks = []
     


    while True:
        print ('inizio frame',nCiclo)
        cur_data = next(gen)
        if cur_data == None:
            break

        img = cur_data['img']
        gtBoxes = [] 

        listObjects = []
        height, width, channels = img.shape

        if img is None:
            break

        if 'feats' in cur_data.keys():
            cur_feats = cur_data['feats']
            currentVector = []
            for feat in cur_feats:
                currentVector.append(feat)
            print ('feature',len(currentVector))

        if nCiclo == 0:
            pastVector = currentVector

        if nCiclo>0:
            matFeat = confrontoFeatures(pastVector,currentVector)
            for j in range(len(matFeat)):
                    #trovo il massimo valore e gli indici di riga e colonna relativi della matrice dei confronti 
                    m,r,c = getMaximumValue(matFeat)
                    #restituisce l indice se vi e un match tra un box gia presente in traccia e il box con indice r
                    k = findTracksF(pastVector[r],listTracksF) 
                    #Match
                    if m > 0.7:
                        # se esiste una traccia attiva l aggiorna
                        if (k > 0) and (listTracksF[k].state == 'active'):
                            F = currentVector[c]
                            listTracksF[k].boxes.append(F)
                            listTracksF[k].sequence.append(c)
                        #altrimenti ne crea una 
                        else:
                            t = Track(currentID,nCiclo)
                            currentID = currentID + 1
                            t.sequence.append(r)
                            t.sequence.append(c)
                            t.boxes.append(pastVector[r])
                            t.boxes.append(currentVector[c])
                            listTracksF.append(t)
                            
                    else: #NO MATCH
                        if (k>0) and (listTracksF[k].state == 'active'):
                           listTracksF[k].delete()

            pastVector = currentVector


      

        #cv2.imshow('img', img)
        cv2.imshow('img',img)
        #cv2.waitKey(int(1000/FPS))
        cv2.waitKey(1)
        print (nCiclo)
        nCiclo = nCiclo+1
        #print 'fine frame'

    print ('fine')
    pickle.dump(listTracksF, open( 'database/listTrackFeatures.db', "wb" ) )

if __name__ == ' __main__ ':
    
    main()

main()


             













