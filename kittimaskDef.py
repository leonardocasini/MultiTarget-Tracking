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
import time

def createMatrixMasks(A,B):
    tmp = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            #Intersection over Union calculate with logical operation
            tmp[i,j] = np.sum(np.logical_and(A[i], B[j])) / np.sum(np.logical_or(A[i], B[j])).astype(np.float)
    return tmp

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
    
def findTracks(mask,listOfTracks):
    found = -1 
    A = mask
    B = listOfTracks[k].boxes[- 1]
    if np.sum(np.logical_and(A[i], B[j])) / np.sum(np.logical_or(A[i], B[j])).astype(np.float) > 0.7:
        found = k
    return found
class Track:
  def __init__(self,id = None,frame = None):

    self.id = id
    self.frame = frame
    self.sequence = []
    self.masks = []
    self.color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
    self.countNoMatch = 0
    self.state = 'new'

  #This function return the index of track in listTrack 
  #if the last mask saved is equal the selected mask
  def delete(self):
    self.countNoMatch = self.countNoMatch + 1 
    if self.countNoMatch > 2:
        self.state = 'death'

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick #.astype("int")

def vis_mask(img, mask,width,height, col, alpha=0.4, show_border=True, border_thick= -1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)
    #np.PredictionBoxes(col)
    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * (400/255.0)

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1,col, border_thick, cv2.LINE_AA)
        #cv2.drawContours(c, contours, -1, 1, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

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
    nma = input("Select the maximum value for countNoMatch value [0,4]? ")
    noMatchAllowed = float(nma)

    #Initialization 
    nIteration = 0
    pastboxes = []
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

        if 'dets' in cur_data.keys():
            dets = cur_data['dets']
            new_bs = []
            bs = non_max_suppression_fast(np.array(dets[0]), 0.8)
            i = 0
            for box, obj_type, score in zip(*dets):
                if (i in bs) and (score > Score_treshHold) and ((obj_type == 'car') or (obj_type == 'truck')):
                    new_bs.append(i)
                i=i+1

        if 'masks' in cur_data.keys():
            currentmasks = []
            i = 0
            masks = cur_data['masks']
            for mask in masks:
                if i in new_bs:
                     currentmasks.append(mask.astype(np.bool))
                i=i+1
            
            #In the first iteration save the currentmasks in the pastmasks
            if nCiclo == 0:
                pastmasks = currentmasks  
            #In next iterations masks are first compared and then updated 
            if nIteration>0:
                #If currentmasks is empty it only update pastmasks with currentmasks
                if len(tmp) > 0:
                    mat = createMatrixMasks(pastmasks,currentmask)
                    #create a list of Tracks index empty 
                    listTrackUpdated = []
                    #iterate until the matrix is not zereos
                    while mat.any(): 
                        #get max value of matrix and them index of row and column
                        m,r,c = getMaximumValue(mat) 
                        #Return index of Track where r element of pastboxes is uploaded
                        k = findTracks(pastmask[r],listTracksM)
                        #if it exists save this index in listTrackUpdated
                        if k >= 0:
                            listTrackUpdated.append(k) 
                        #Match
                        if m > Iou_Treshold:
                            #Uploading of track If it exist and them state is active or new
                            if (k >= 0) and ((listTracks[k].state == 'active') or (listTracks[k].state == 'new')):
                                if (listTracks[k].state == 'dying'):
                                    listTracks[k].countNoMatch = 0
                                mask = currentmasks[c].astype(np.uint8)
                                listTracks[k].state = 'active'
                                listTracks[k].masks.append(mask)
                                listTracks[k].sequence.append(c)
                                t.sequence.append(c)
                                img = vis_mask(img, mask,width,height, listTracks[k].color)
                            else:
                                t = Track(currentID,nCiclo)
                                currentID = currentID + 1
                                box = currentmasks[c].astype(np.uint8)
                                t.masks.append(pastmasks[r])
                                t.masks.append(box)
                                t.sequence.append(r)
                                t.sequence.append(c)
                                listTracks.append(t)
                                img = vis_mask(img, mask ,width,height,t.color)
           
                    #A method called delete is called for active tracks that have not been updated 
                    for i in range(len(listTracks)):
                        if (i not in listTrackUpdated) and (listTracks[i].state == 'active'):
                            mask = listTracks[i].maks[-1]
                            img = vis_mask(img, mask ,width,height,listTracks[i].color)
                            listTracks[i].delete()

            pastmasks = currentmasks
        cv2.imshow('img',img)
        cv2.waitKey(1)
        nIteration = nIteration+1
    #All tracks calculated are saved into a database
    pickle.dump(listTracksM, open( 'database/listMask' + video_id +'.db', "wb" ) )

if __name__ == ' __main__ ':
    main()

#For starting program remove hashtag from main()
#main()
