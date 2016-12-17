"""
The interface of DetectionList, the format of *.lst file like as:
 num_class:1
classes:face
img_path1 num_box, box1_x, box1_y, box1_w, box1_h,box2_x, box2_y, box2_w, box2_h...
img_path2 num_box, box1_x, box1_y, box1_w, box1_h,box2_x, box2_y, box2_w, box2_h, box1_h,box2_x, box2_y, box2_w, box2_h......
...
in wihich img_path is the relative path
"""

import os
import numpy as np
import scipy.sparse
import scipy.io
import cPickle
from imdb import IMDB
from voc_eval import voc_eval
from helper.processing.bbox_process import unique_boxes, filter_small_boxes

class DetectionList(IMDB):
    def __init__(self, dataset_name, list_file, dataset_root, outdata_path):
        """
        fill basic information to initialize imdb
        :param dataset_name: the name of your dataset
        :param list_file: train or val or trainval
        :param dataset_root: the root path of your dataset
        :param outdata_path: 'selective_search_data' and 'cache'
        :return: imdb object
        """
        super(DetectionList, self).__init__(dataset_name)  # set self.name
        self.dataset_name = dataset_name
        self.list_file = list_file
        self.dataset_root = dataset_root
        self.outdata_path = outdata_path

        self.f_list = open(self.list_file, 'r')
        line = self.f_list.readline().strip('\n').split(':')
        assert(line[0] == "num_class"), "fisrt line should be: num_clss:XX"
        self.num_classes = int(line[1]) + 1  # consider background

        line = self.f_list.readline().strip('\n').split(':')
        assert(line[0] == "classes"), "second line should be: classes:XX1 XX2 XX3..."
        self.classes = ['__background__'] + line[1:self.num_classes+1]

        self.annos = [x.strip('\n').split(' ') for x in self.f_list.readlines()]
        self.num_images = len(self.annos)  # no need -2
        self.image_set_index = range(self.num_images)
        self.f_list.close()

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.outdata_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.dataset_root, self.annos[index][0])
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from annotation list file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        num_objs = int(self.annos[index][1])
        assert num_objs > 0

        boxes = np.zeros((num_objs, 4), dtype=np.int16)  # no uint16 because of the coord which out of range
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix in range(num_objs):
            x, y, w, h = self.annos[index][2 + 4*ix : 2 + 4*ix + 4]
            # be careful that pixel indexes should be 0-based
            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(w) - 1.0
            y2 = y1 + float(h) - 1.0
            if x2 - x1 <= 0:  # prevent illegal label
                x2 = x1 + 2
            if y2 - y1 <= 0:
                y2 = y1 + 2
            if self.num_classes == 2:
                cls = 1
            else:
                NotImplemented  # TODO(support multi object detection)
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}