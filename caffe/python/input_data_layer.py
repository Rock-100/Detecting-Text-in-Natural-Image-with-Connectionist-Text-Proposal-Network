from cfg import Config as cfg
import numpy as np
import yaml, caffe
from other import clip_boxes
from anchor import AnchorText


class InputDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._train_images = layer_params['train_images']
        self.anchor_generator=AnchorText()
        self._num_anchors = self.anchor_generator.anchor_num

        print "self._train_images", self._train_images
        # data
        top[0].reshape(1, 3, cfg.scale, cfg.scale)
        # im_info
        top[1].reshape(1, 2)
        # gt_boxes
        top[1].reshape(1, 4)

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
