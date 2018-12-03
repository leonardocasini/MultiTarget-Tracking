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

def createMatrix(A,B):
    currentmask = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            currentmask[i,j] = cosine_similarity(A[i],B[j])
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
    

def findTracks(vector,listOfTracks):
    found = -1 
    for k in range(len(listOfTracks)):
        if np.array_equal(vector,listOfTracks[k].boxes[len(listOfTracksF[k].boxes) - 1]) > 0.7:
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
    self.state = 'dying'
    if self.countNoMatch > self.nma: 
        self.state = 'death'

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)


def main():
    #Change the path for your dataset
    dataset_path = '/Users/andreasimioni/Desktop/kitti/kitti5'
    #Select id of video
    video_id = '0004'
    file_path = dataset_path + '/label_02/' + video_id + '.txt'
    img_path = dataset_path + '/image_02/' + video_id
    
    annot = KittiAnnotation(file_path, img_path)
    data_from_generator = ('img', 'annot', 'dets','masks')
    gen = annot.annot_generator(data=data_from_generator, loop=False)

    #Parameters of analisys
    It = input("Select a theshHold value for the IoU [0,1]? ")
    Iou_Treshold = float(It)
    st = input("Select a theshHold value for the score of prediction boxes [0,1]? ")
    Score_treshHold = float(st)
    nma = input("Select the maximum value for countNoMatch value? ")
    noMatchAllowed = float(nma)
    
    #Initialization 
    nIteration = 0
    pastVectors = []
    listTracks = []
    currentID = 0
     
    while True:
        cur_data = next(gen)
        if cur_data == None:
            break

        img = cur_data['img']
        height, width, channels = img.shape

        if img is None:
            break

        if 'feats' in cur_data.keys():
            cur_feats = cur_data['feats']
            currentVectors = []
            for feat in cur_feats:
                currentVectors.append(feat)

        if nIteration == 0:
            pastVectors = currentVectors

        if nIteration>0:
            matFeat = createMatrix(pastVectors,currentVectors)
            while mat.any():
                    m,r,c = getMaximumValue(matFeat)
                    k = findTracks(pastVectors[r],listTracks) 
                    #Match
                    if m > 0.7:
                        if (k > 0) and (listTracks[k].state == 'active'):
                            if (listTracks[k].state == 'dying'):
                                    listTracks[k].countNoMatch = 0
                            F = currentVectors[c]
                            listTracks[k].boxes.append(F)
                            listTracks[k].sequence.append(c)
                        else:
                            t = Track(currentID,nIteration)
                            currentID = currentID + 1
                            t.sequence.append(r)
                            t.sequence.append(c)
                            t.boxes.append(pastVectors[r])
                            t.boxes.append(currentVectors[c])
                            listTracks.append(t)
            #A method called delete is called for active tracks that have not been updated 
            for i in range(len(listTracks)):
                if (i not in listTrackUpdated) and (listTracks[i].state == 'active') or (listTracks[i].state == 'dying'):
                    listTracks[i].delete()
            pastVectors = currentVectors

        cv2.imshow('img',img)
        cv2.waitKey(1)
        nIteration = nIteration+1

    pickle.dump(listTracks, open( 'database/listTrackFeatures.db', "wb" ) )

if __name__ == ' __main__ ':
    main()

#For starting program remove hashtag from main()
#main()
