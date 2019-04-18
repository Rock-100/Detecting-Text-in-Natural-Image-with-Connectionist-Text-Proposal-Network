from cfg import Config as cfg
from other import prepare_img, normalize
import numpy as np
from utils.cpu_nms import cpu_nms as nms
from text_proposal_connector import TextProposalConnector


class TextProposalDetector:
    """
        Detect text proposals in an image
    """
    def __init__(self, caffe_model):
        self.caffe_model=caffe_model

    def detect(self, im, mean):
        im_data=prepare_img(im, mean)
        _=self.caffe_model.forward2({
            "data": im_data[np.newaxis, :],
            "im_info": np.array([[im_data.shape[1], im_data.shape[2]]], np.float32)
        })
        rois=self.caffe_model.blob("rois")
        scores=self.caffe_model.blob("scores")
        sides=self.caffe_model.blob("sides")
	#self.caffe_model.print_blobs_shape()
        return rois, scores, sides


class TextDetector:
    """
        Detect text from an image
    """
    def __init__(self, text_proposal_detector):
        self.text_proposal_detector=text_proposal_detector
        self.text_proposal_connector=TextProposalConnector()

    def detect(self, im, sq = False):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        """
        text_proposals, scores, sides=self.text_proposal_detector.detect(im, cfg.MEAN)
   
        keep_inds=np.where(scores>cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores, sides=text_proposals[keep_inds], scores[keep_inds], sides[keep_inds]
        

        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores, sides=text_proposals[sorted_indices], scores[sorted_indices], sides[sorted_indices]

        # nms for text proposals
        keep_inds=nms(np.hstack((text_proposals, scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores, sides=text_proposals[keep_inds], scores[keep_inds], sides[keep_inds]

	if sq:
	    return text_proposals
        scores=normalize(scores)

        text_lines=self.text_proposal_connector.get_text_lines(text_proposals, scores, im.shape[:2], sides)

        keep_inds=self.filter_boxes(text_lines)
        text_lines=text_lines[keep_inds]

        # nms for text lines
        if text_lines.shape[0]!=0:
            keep_inds=nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
            text_lines=text_lines[keep_inds]

        return text_lines

    def filter_boxes(self, boxes):
        heights=boxes[:, 3]-boxes[:, 1]+1
        widths=boxes[:, 2]-boxes[:, 0]+1
        scores=boxes[:, -1]
        return np.where((widths/heights>cfg.MIN_RATIO) & (scores>cfg.LINE_MIN_SCORE) &
                          (widths>(cfg.TEXT_PROPOSALS_WIDTH*cfg.MIN_NUM_PROPOSALS)))[0]
