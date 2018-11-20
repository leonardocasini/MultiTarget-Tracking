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

acc = mm.MOTAccumulator(auto_id=True)
def getBestIou(G,Arrays):
    selectedBox = None
    bestIou = 0
    for k in range(len(Arrays)):
            B = Arrays[k]['img']
            Iou = getIntersectionOverUnion(G,B)
            if Iou > bestIou:
                bestIou = Ioub
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
    tmp = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            tmp[i,j] = bb_intersection_over_union(A[i],B[j])
    return tmp

def confrontoMask(A,B):
    tmp = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            tmp[i,j] = np.sum(np.logical_and(A[i], B[j])) / np.sum(np.logical_or(A[i], B[j])).astype(np.float)
            print("AOOOOOOOOO",tmp)
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
    

def findTracks(box,listOfTracks):
    #print box, listOfTracks
    found = -1 
    for k in range(len(listOfTracks)):
        #print listTracks[k].boxes[len(listTracks[k].boxes) - 1]
        if box == listTracks[k].boxes[len(listTracks[k].boxes) - 1]:
            found = k

    return found 

def findTracksM(box,listOfTracksM):
    #print box, listOfTracks
    found = -1 
    for k in range(len(listOfTracksM)):
        if (box == listOfTracksM[k].boxes[len(listOfTracksM[k].boxes) - 1]).all():
            found = k

    return found 


class Track:
  def __init__(self,id = None,frame = None):

    self.id = id
    self.frame = frame
    self.sequence = []
    self.boxes = []
    self.color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
    self.countNoMatch = 0
    self.state = 'new'

  def delete(self):
    self.countNoMatch = self.countNoMatch + 1 
    if self.countNoMatch > 2:
        print('traccia',self.id,'morta')
        self.state = 'death'






            
 #def getColor(track_id):

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



def get_colormap(N):
    cm = plt.cm.get_cmap(None, N)
    colors = [cm(x) for x in range(N)]
    for i in range(N):
        colors[i] = [x * 255 for x in colors[i][:3]]
    # randomize colormap
    shuffle(colors)
    return colors

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

    dataset_path = '/Users/andreasimioni/Desktop/kitti/kitti5'
    video_id = '0004'
    file_path = dataset_path + '/label_02/' + video_id + '.txt'
    img_path = dataset_path + '/image_02/' + video_id
    annot = KittiAnnotation(file_path, img_path)
    listTracks = []
    listTracksM  = []
    groundTracks = [] 
    Iou_Treshold = 0.3 
    Score_treshHold = 0.8

     


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

    currentIDMask = 0


    

    while True:
        print ('inizio frame',nCiclo)
        cur_data = next(gen)
        if cur_data == None:
            break

        img = cur_data['img']


        listObjects = []
        height, width, channels = img.shape

        if img is None:
            break

        # annot
        if 'annot' in cur_data.keys():
            annot = cur_data['annot']
            tmp = []
            cur_gt_track_ids = []
            for track_id in annot.keys():
                box, obj_type = annot[track_id]
                # PredictionBoxes.append([box,track_id])
                if track_id not in listObjects:
                    listObjects.append(track_id) 
                #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), 1, thickness=4)

                a = np.zeros((height,width))
                #cv2.rectangle(a,(box[0], box[1]), (box[2], box[3]),1, thickness = -1)
                #cv2.putText(img,track_id,(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors_tracks[int(track_id)], 2)
                tmp.append({'track_id': track_id, 'box': box}) 
                #obj_type

        # dets

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
            tmp = []
            i = 0
            masks = cur_data['masks']
            #print('EEEEEEEEOOOOOOOOOOOOOOOO', masks)
            for mask in masks:
                #img,c = vis_mask(img, mask,width,height, (0, 0, 255))
                if i in new_bs:
                     print(mask.astype(np.bool))
                     tmp.append(mask.astype(np.bool))
                i=i+1
            
            if nCiclo == 0:
                #assegnazione delle mask del frame precedente al passo zero
                pastmask = tmp  
            if nCiclo>0:
                #confronto tra i box del frame precedente e quelli correnti
                
                if len(tmp) > 0:
                    mat = confrontoMask(pastmask,tmp)
                    
                    listTrackUpdated = []
                    
                    while mat.any(): 
                        #trovo il massimo valore e gli indici di riga e colonna relativi della matrice dei confronti 
                        m,r,c = getMaximumValue(mat) 
                        #restituisce l indice se vi e un match tra un box gia presente in traccia e il box con indice r
                        k = findTracksM(pastmask[r],listTracksM)
                        if k >= 0:
                            listTrackUpdated.append(k) 
                        #Match
                        if m > Iou_Treshold:
                            
                            # se esiste una traccia attiva l aggiorna
                            if (k >= 0) and ((listTracksM[k].state == 'active') or (listTracksM[k].state == 'new')):
                                box = tmp[c].astype(np.uint8)
                                listTracksM[k].boxes.append(box)
                                t.sequence.append(c)
                                img = vis_mask(img, box,width,height, listTracksM[k].color)
                                #cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]),listTracks[k].color, thickness = 3)
                                #cv2.putText(img,str(listTracks[k].id),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, listTracks[k].color, 2)
                            #altrimenti ne crea una 
                            else:
                                t = Track(currentIDMask,nCiclo)
                                currentIDMask = currentIDMask + 1
                                box = tmp[c].astype(np.uint8)
                                t.boxes.append(pastmask[r])
                                t.boxes.append(box)
                                t.sequence.append(r)
                                t.sequence.append(c)
                                listTracksM.append(t)
                                img = vis_mask(img, box,width,height,t.color)
                                #cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]),t.color, thickness = 3)
                            #cv2.putText(img,str(currentID-1),(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors_tracks[int(currentID-1)], 2)
                        

                    for i in range(len(listTracks)):
                            if (i not in listTrackUpdated) and (listTracks[i].state == 'active'):
                                listTracks[i].delete()


            pastmask = tmp
        

            # if nCiclo>0:
            #     tracker_active_tracks_ids = []
            #     tracker_prev_boxes  = []
            #     for i in range(len(listTracksM)):
            #         if listTracksM[i].state == 'active' or listTracksM[i] == 'new':
            #             print('id traccia attiva',listTracksM[i].id, listTracksM[i].state
            #                 )
            #             tracker_prev_boxes.append(listTracksM[i].boxes[len(listTracksM[i].boxes)-1])
            #             tracker_active_tracks_ids.append((listTracksM[i].id))

            #     predictboxes_for_metrics = []
            #     for i in range(len(tmp)):
            #         predictboxes_for_metrics.append(tmp[i]['box'])

            #     dist_mat = mm.distances.iou_matrix(predictboxes_for_metrics, tracker_prev_boxes, max_iou= 1)
            #     acc.update(cur_gt_track_ids, tracker_active_tracks_ids, dist_mat)



        cv2.imshow('img',img)
        cv2.waitKey(1)
        nCiclo = nCiclo+1
        #pickle.dump( listTracksM,open('database/listTrackPredicted' + video_id +'.db', "wb" ))
        #print 'fine frame'
    print ('fine')
    print (nCiclo)
    pickle.dump(listTracksM, open( 'database/listMask' + video_id +'.db', "wb" ) )
if __name__ == ' __main__ ':
    main()




main()





         













