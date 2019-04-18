# -*- coding: utf-8 -*-
"""
@author: Jack
"""

from cfg import Config as cfg
import cv2, os, sys
import numpy as np
import yaml, caffe
from other import prepare_img, resize_im, inside_image
import random

def get_img_name(root_dir, img_file_list):
    file_list = os.listdir(root_dir)
    for f in file_list:
        if '.' in f:
            img_file_list.append(root_dir + '/' + f)
        else:
            get_img_name(root_dir + '/' + f, img_file_list)

class InputDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self._train_images = layer_params['train_images']
        self._imnames = []
        get_img_name(self._train_images, self._imnames)
        self._imnames = [x for x in self._imnames if x.split('.')[-1] != 'txt']
        random.shuffle(self._imnames)
        self._image_index = 0

        # data
        top[0].reshape(1, 3, cfg.SCALE, cfg.SCALE)
        # im_info
        top[1].reshape(1, 2)
        # gt_boxes
        top[2].reshape(1, 4)
        # side_pos
        top[3].reshape(1, 1)

    def forward(self, bottom, top):
        while True:
            name = self._imnames[self._image_index].split('.')[0]
            self._image_index += 1
            if self._image_index == len(self._imnames):
                self._image_index = 0

            image = cv2.imread(name + '.jpg')
            image, scale = resize_im(image, cfg.SCALE, cfg.MAX_SCALE)
            data = prepare_img(image, cfg.MEAN)
            im_info = np.array([[data.shape[1], data.shape[2]]], np.float32)
            data = data[np.newaxis, :]

            gt_boxes_list = []
            with open(name + '.txt', 'r') as f:
                for line in f:
                    line_data = line.split(',')
                    gt_boxes_list.append([int(line_data[0]), int(line_data[1]), int(line_data[2]), int(line_data[3])])
            gt_boxes_list = [[int(x * scale) for x in box] for box in gt_boxes_list]

            divide_gt_boxes_list = []
            side_pos = []
            for box in gt_boxes_list:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2] + box[0] - 1
                y2 = box[3] + box[1] - 1
                if y2 - y1 + 1 <= cfg.TEXT_PROPOSALS_WIDTH / 2:
                    continue
                if x2 - x1 + 1 < cfg.TEXT_PROPOSALS_WIDTH:
                    continue
                start = x1
                while start % cfg.TEXT_PROPOSALS_WIDTH != 0:
                    start += 1
                end = x2 + 1
                while end % cfg.TEXT_PROPOSALS_WIDTH != 0:
                    end -= 1
                begin_flag = 1
                tmp_side_pos = []
                while start < end:
                    if inside_image(start, y1, im_info[0, :]) and inside_image(start + cfg.TEXT_PROPOSALS_WIDTH - 1, y2, im_info[0, :]):
                        divide_gt_boxes_list.append([start, y1, start + cfg.TEXT_PROPOSALS_WIDTH - 1, y2])
                        if begin_flag:
                            begin_flag = 0
                            tmp_side_pos.append(x1)
                        else:
                            if start + cfg.TEXT_PROPOSALS_WIDTH == end:
                                tmp_side_pos.append(x2)
                            else:
                                tmp_side_pos.append(-1)
                    start += cfg.TEXT_PROPOSALS_WIDTH
                for p in tmp_side_pos:
                    side_pos.append(p)
            gt_boxes = np.array(divide_gt_boxes_list)
            side_pos = np.array(side_pos)
            if len(divide_gt_boxes_list):
                break
        
        top[0].reshape(*(data.shape))
        top[0].data[...] = data.astype(np.float32, copy=False)

        top[1].reshape(*(im_info.shape))
        top[1].data[...] = im_info.astype(np.float32, copy=False)

        top[2].reshape(*(gt_boxes.shape))
        top[2].data[...] = gt_boxes.astype(np.float32, copy=False)

        top[3].reshape(*(side_pos.shape))
        top[3].data[...] = side_pos.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
