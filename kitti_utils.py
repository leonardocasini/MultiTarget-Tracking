import csv
import numpy as np
import ntpath
from glob import glob
import cv2
import pickle
import gzip


def load(filename):
    """Loads a compressed object from disk
    """
    fp=gzip.open(filename,'rb') # This assumes that primes.data is already packed with gzip
    object=pickle.load(fp, encoding='latin1')
 
    # file = gzip.open(filename, 'r')
    # buffer = ""
    # while True:
    #     data = file.read()
    #     if data == "":
    #         break
    #     buffer += data
    # object = pickle.load(buffer, encoding='latin1')
    # file.close()
    return object


class KittiAnnotation:
    """
    Read the kitti annotation file
    frame_ids, track_ids, obj_type, truncation, occlusion, obs_angle, x1, y1, x2, y2, w, h, l, X, Y, Z, yaw
    """

    def __init__(self, filepath, imgpath):
        # read frames
        self.img_paths = sorted(glob(imgpath + '/*.png'))

        # read annotations
        with open(filepath, 'r') as f: #open('sample.csv', "rt", encoding=<theencodingofthefile>)
            reader = csv.reader(f)
            l = list(reader)
        self.annotations = [x[0].split() for x in l]

        frame_ids = [int(x[0]) for x in self.annotations]
        track_ids = [int(x[1]) for x in self.annotations]
        self.num_frames = len(self.img_paths)
        self.num_tracks = max(track_ids) + 1
        self.objects_per_frame = [np.where(frame_ids == x) for x in np.unique(frame_ids)]
        self.annot_name = ntpath.basename(filepath)
        self.boxes_per_frame = [{} for x in range(self.num_frames)]
        for n, a in enumerate(self.annotations):
            if int(a[1]) >= 0:
                self.boxes_per_frame[int(a[0])][a[1]] = ([int(a[6].split('.')[0]),
                                                          int(a[7].split('.')[0]),
                                                          int(a[8].split('.')[0]),
                                                          int(a[9].split('.')[0])], a[2])

    def __str__(self):
        return 'track ' + self.annot_name

    def annot_generator(self, data=('img'), loop=False):
        """
        :param data: list of data types to return ('img', 'annot')
        :param loop: set to True to loop over the video indefinitely
        :return: generator over video frames
        """

        is_looping = True
        while is_looping:
            for f in range(self.num_frames):
                p = self.img_paths[f].replace('image_02', 'masks_and_dets') + '.pkl'

                mask_and_dets = load(p)
                data_to_yield = {}
                if 'img' in data:
                    cur_img = cv2.imread(self.img_paths[f])
                    data_to_yield['img'] = cur_img
                if 'annot' in data:
                    data_to_yield['annot'] = self.boxes_per_frame[f]
                if 'dets' in data:
                    data_to_yield['dets'] = ([x['bbox'] for x in mask_and_dets],
                                             [x['class_name'] for x in mask_and_dets],
                                             [x['score'] for x in mask_and_dets])
                if 'masks' in data:
                    data_to_yield['masks'] = [x['mask'] for x in mask_and_dets]
                if 'feats' in data:
                    data_to_yield['feats'] = [x['feats'] for x in mask_and_dets]
                yield data_to_yield
            if not loop:
                is_looping = False
            else:
                print ('Starting again from the beginning...')
        # we have reached the end of the video

        yield None
        #yield [None] * len(data)
