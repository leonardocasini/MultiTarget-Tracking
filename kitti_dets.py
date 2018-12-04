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

def createMatrix(A,B):
    tmp = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            tmp[i,j] = bb_intersection_over_union(A[i],B[j])
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
    

#This function return the index of track in listTrack 
#if the last box saved is similar to selected box
def findTracks(box,listOfTracks,):
    found = -1 
    for k in range(len(listOfTracks)):
        if bb_intersection_over_union(box,listOfTracks[k].boxes[len(listOfTracks[k].boxes) - 1]) > 0.7:
            found = k 
    return found 


class Track:
  def __init__(self,id = None, frame = None,noMatchAllowed=None):
    self.id = id
    self.boxes = []
    self.sequence = []
    self.color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
    self.countNoMatch = 0
    self.startingFrame = frame
    self.state = 'new'
    self.nma = noMatchAllowed
  #set track's state to death if countNoMatch is greater than a treshold
  def delete(self):
    self.countNoMatch = self.countNoMatch + 1 
    self.state = 'dying'
    if self.countNoMatch > self.nma: 
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
    pastboxes = []
    listTracks = []
    currentID = 0
    acc = mm.MOTAccumulator(auto_id=True)

    while True:
        cur_data = next(gen)
        if cur_data == None:
            break

        img = cur_data['img']

        height, width, channels = img.shape

        if img is None:
            break

        if 'annot' in cur_data.keys():
            annot = cur_data['annot']
            tmp = []
            cur_gt_track_ids = []

            for track_id in annot.keys():
                box, obj_type = annot[track_id]
                if ((obj_type == 'Car') or (obj_type == 'Van')):
                    cur_gt_track_ids.append(track_id)
                    a = np.zeros((height,width))
                    tmp.append({'track_id': track_id, 'box': box})
 
        if 'dets' in cur_data.keys():
            dets = cur_data['dets']
            currentboxes = []
            #bs contain list of index
            bs = non_max_suppression_fast(np.array(dets[0]), 0.8)
            i = 0
            for box, obj_type, score in zip(*dets):
                #Filtering box with score, bs and obj_type
                if (i in bs) and (score > Score_treshHold) and ((obj_type == 'car') or (obj_type == 'truck')):
                    #Resulting Boxes are loaded in currentboxes
                    currentboxes.append([box[0],box[1],box[2],box[3]])
                i=i+1

        #In the first iteration save the currentboxes in the pastboxes 
        if nIteration == 0:
            pastboxes = currentboxes  
        #In next iterations the boxes are first compared and then updated 
        if nIteration>0:
                #If currentboxes is empty it only update pastboxes with currentboxes
                if len(currentboxes) > 0:
                    mat = createMatrix(pastboxes,currentboxes)
                    #create a list of Tracks index empty 
                    listTrackUpdated = []
                    #iterate until the matrix is not zereos
                    while mat.any():
                        #get max value of matrix and them index of row and column
                        m,r,c = getMaximumValue(mat) 
                        #Return index of Track where r element of pastboxes is uploaded
                        k = findTracks(pastboxes[r],listTracks)
                        #if it exists save this index in listTrackUpdated
                        if k >= 0:
                            listTrackUpdated.append(k)
                        
                        if m > Iou_Treshold:
                            #Uploading of track If it exist and them state is active or new
                            if (k >= 0) and ((listTracks[k].state == 'active') or (listTracks[k].state == 'new')) :
                                if (listTracks[k].state == 'dying'):
                                    listTracks[k].countNoMatch = 0
                                box = currentboxes[c]
                                listTracks[k].state = 'active'
                                listTracks[k].boxes.append(box)
                                listTracks[k].sequence.append(c)
                                #Draw box uploaded
                                cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]),listTracks[k].color, thickness = 3)
                                cv2.putText(img,str(listTracks[k].id),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, listTracks[k].color, 2)
                            #In other case a new track is created 
                            else:
                                t = Track(currentID,nIteration,noMatchAllowed)
                                currentID = currentID + 1
                                box = currentboxes[c]
                                #During creation both of boxesis saved
                                t.boxes.append(pastboxes[r])
                                t.boxes.append(currentboxes[c])
                                #The same for index
                                t.sequence.append(r)
                                t.sequence.append(c)
                                #Once created it is added to the tracks 
                                listTracks.append(t)
                                #Draw currentbox
                                cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]),t.color, thickness = 3)
                                cv2.putText(img,str(currentID-1),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, t.color, 2)

                    #A method called delete is called for active tracks that have not been updated 
                    for i in range(len(listTracks)):
                        if (i not in listTrackUpdated) and (listTracks[i].state == 'active') or (listTracks[i].state == 'dying'):
                            box = listTracks[i].boxes[-1]
                            cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]),listTracks[i].color, thickness = 6)
                            cv2.putText(img,str(listTracks[i].id),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, listTracks[i].color, 2)
                            listTracks[i].delete()
                pastboxes = currentboxes
        
        if nIteration>0:
            #List of Track id with state active or new 
            tracker_active_tracks_ids = []
            #Array of the last box saved in track 
            tracker_prev_boxes  = []
            for i in range(len(listTracks)):
                if listTracks[i].state == 'active' or listTracks[i] == 'new':
                    tracker_prev_boxes.append(listTracks[i].boxes[-1])
                    tracker_active_tracks_ids.append((listTracks[i].id))
            #Array of ground truth boxes
            predictboxes_for_metrics = []
            for i in range(len(tmp)):
                predictboxes_for_metrics.append(tmp[i]['box'])
            #Matrix of distance between ground truth boxes and track boxes
            dist_mat = mm.distances.iou_matrix(predictboxes_for_metrics, tracker_prev_boxes, max_iou= 1)
            #Updating acc
            acc.update(cur_gt_track_ids, tracker_active_tracks_ids, dist_mat)

        cv2.imshow('img',img) 
        cv2.waitKey(1)
        nIteration = nIteration+1

    #All tracks calculated are saved into a database
    pickle.dump(listTracks, open( 'database/listDetections' + video_id +'.db', "wb" ) )
    #Calculate metrics of accuracy
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp',
                'num_fragmentations', 'num_switches', 'num_false_positives',
                'num_misses', 'mostly_tracked', 'partially_tracked',
                'mostly_lost', 'precision', 'recall'], name='acc')
    #Save this metrics into a file csv with video_id and the parameters of evaluations('Iou_Treshold','Score_treshHold')  
    with open('table_results.csv', 'a') as csvfile:
        fieldnames = ['videoID','Iou_Treshold','Score_treshHold','num_frames', 'mota', 'motp',
                'num_fragmentations', 'num_switches', 'num_false_positives',
                'num_misses', 'mostly_tracked', 'partially_tracked',
                'mostly_lost', 'precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'videoID':video_id,'Iou_Treshold':Iou_Treshold,'Score_treshHold':Score_treshHold,'num_frames':summary['num_frames']['acc'], 'mota': summary['mota']['acc'],'motp':summary['motp']['acc'],
            'num_fragmentations':summary['num_fragmentations']['acc'],'num_switches':summary['num_switches']['acc'],
            'num_false_positives':summary['num_false_positives']['acc'],'num_misses':summary['num_misses']['acc'],
            'mostly_tracked':summary['mostly_tracked']['acc'],'partially_tracked':summary['partially_tracked']['acc'],'mostly_lost':summary['mostly_lost']['acc'],
            'precision':summary['precision']['acc'],'recall': summary['recall']['acc']})     

if __name__ == ' __main__ ':
    main()

#For starting program remove hashtag from main()
main()




