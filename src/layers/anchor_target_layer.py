# -*- coding: utf-8 -*-
"""
@author: Jack
"""

import os
import caffe
import yaml
from cfg import Config as cfg
import numpy as np
import numpy.random as npr
from anchor import AnchorText
from utils.bbox import bbox_overlaps

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._feat_stride = layer_params['feat_stride']
        self.anchor_generator=AnchorText()
        self._num_anchors = self.anchor_generator.anchor_num
        
        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 2, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 2, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 2, height, width)
        # sr_targets
        top[4].reshape(1, A, height, width)
        # sr_inside_weights
        top[5].reshape(1, A, height, width)
        # sr_outside_weights
        top[6].reshape(1, A, height, width)

    def forward(self, bottom, top):

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]
        # side_pos
        side_pos = bottom[3].data


        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes'
            print  gt_boxes
            print 'rpn: side_pos.shape', side_pos.shape
            print 'rpn: side_pos'
            print  side_pos

        A = self._num_anchors
        all_anchors = self.anchor_generator.locate_anchors((height, width), self._feat_stride)
        total_anchors = all_anchors.shape[0]

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= 0) &
            (all_anchors[:, 1] >= 0) &
            (all_anchors[:, 2] < im_info[1]) &  # width
            (all_anchors[:, 3] < im_info[0])    # height
        )[0]
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inside_anchors', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        init_gt_argmax_overlaps = gt_argmax_overlaps
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]	

        if DEBUG:
            print "overlaps shape", overlaps.shape
            print "argmax_overlaps shape", argmax_overlaps.shape
            print "gt_argmax_overlaps shape", gt_argmax_overlaps.shape
            print "init_gt_argmax_overlaps shape", init_gt_argmax_overlaps.shape
            print "init_gt_argmax_overlaps"	
            print init_gt_argmax_overlaps
            print "max overlaps anchors"
            print anchors[init_gt_argmax_overlaps]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.TRAIN_RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN_RPN_POSITIVE_OVERLAP] = 1

        if DEBUG:
            print "before sample"
            print "positive anchor num", np.sum(labels == 1)
            print "negative anchor num", np.sum(labels == 0)

        # sample positive labels if we have too many
        num_fg = int(cfg.TRAIN_RPN_FG_FRACTION * cfg.TRAIN_RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # sample negative labels if we have too many
        num_bg = cfg.TRAIN_RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
        if DEBUG:
            print "after sample"
            print "positive anchor num", np.sum(labels == 1)
            print "positive anchor", np.where(labels == 1)[0]
            print "negative anchor num", np.sum(labels == 0)

        bbox_targets = np.zeros((len(inds_inside), 2), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array([1, 1])

        bbox_outside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
        bbox_outside_weights[labels == 1, :] = np.array([1, 1])

        if DEBUG:
            print "before map:"
            print "labels.shape", labels.shape
            print "bbox_targets.shape", bbox_targets.shape
            print "bbox_inside_weights.shape", bbox_inside_weights.shape
            print "bbox_outside_weights.shape", bbox_outside_weights.shape

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)	

        max_anchor_inds = inds_inside[init_gt_argmax_overlaps]
        if DEBUG:
            print "max anchors"
            print all_anchors[max_anchor_inds]

        sr_targets = np.empty((total_anchors, ), dtype=np.float32)
        sr_targets.fill(0)
        
        sr_anchor_inds = []
        for i in range(len(side_pos)):
            if side_pos[i] < 0:
                continue
            inds = max_anchor_inds[i]
            side = side_pos[i]
            line_num = int(inds) / int(10 * width)
            for x in [-10, 0, 10]:
                tmp_inds = inds + x
                tmp_line_num = int(tmp_inds) / int(10 * width)
                if tmp_line_num == line_num:
                    center = (all_anchors[tmp_inds][0] + all_anchors[tmp_inds][2]) / 2.0
                    if abs(center - side) > cfg.TRAIN_SIDE_REFINE_MAX:
                        continue
                    sr_anchor_inds.append(tmp_inds)
                    sr_targets[tmp_inds] = (side - center) / cfg.TEXT_PROPOSALS_WIDTH
               
        sr_anchor_inds = [x for x in sr_anchor_inds if sr_anchor_inds.count(x) == 1]
        if len(sr_anchor_inds) > cfg.TRAIN_SR_BATCH:
            sr_anchor_inds=npr.choice(sr_anchor_inds, size=(cfg.TRAIN_SR_BATCH), replace=False)

        sr_inside_weights = np.empty((total_anchors, ), dtype=np.float32)
        sr_inside_weights.fill(0)
        sr_inside_weights[sr_anchor_inds] = 1
        sr_outside_weights = np.empty((total_anchors, ), dtype=np.float32)
        sr_outside_weights.fill(0)
        sr_outside_weights[sr_anchor_inds] = 1

        if DEBUG:
            print "after map:"
            print "labels.shape", labels.shape
            print "bbox_targets.shape", bbox_targets.shape
            print "bbox_inside_weights.shape", bbox_inside_weights.shape
            print "bbox_outside_weights.shape", bbox_outside_weights.shape
            print "sr_targets.shape", sr_targets.shape
            print "sr_inside_weights.shape", sr_inside_weights.shape
            print "sr_outside_weights.shape", sr_outside_weights.shape
            print "side refinement:"
            print "sr_anchor_inds", sr_anchor_inds
            print "sr_anchor", all_anchors[sr_anchor_inds]
            print "sr_targets", sr_targets[sr_anchor_inds]
            print "sr_inside_weights", sr_inside_weights[sr_anchor_inds]
            print "sr_outside_weights", sr_outside_weights[sr_anchor_inds]        

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 2)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 2)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 2)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

        # sr_targets
        sr_targets = sr_targets \
            .reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        top[4].reshape(*sr_targets.shape)
        top[4].data[...] = sr_targets

        # sr_inside_weights
        sr_inside_weights = sr_inside_weights \
            .reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        assert sr_inside_weights.shape[2] == height
        assert sr_inside_weights.shape[3] == width
        top[5].reshape(*sr_inside_weights.shape)
        top[5].data[...] = sr_inside_weights

        # sr_outside_weights
        sr_outside_weights = sr_outside_weights \
            .reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        assert sr_outside_weights.shape[2] == height
        assert sr_outside_weights.shape[3] == width
        top[6].reshape(*sr_outside_weights.shape)
        top[6].data[...] = sr_outside_weights


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def bbox_transform(ex_rois, gt_rois):
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dy, targets_dh)).transpose()
    return targets

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
