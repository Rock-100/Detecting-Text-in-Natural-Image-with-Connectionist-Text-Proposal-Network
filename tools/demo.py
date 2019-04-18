#
# The codes are used for implementing CTPN for scene text detection, described in:
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
#
# These demo codes (with our trained model) are for text-line detection (without
# side-refiement part).
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
#
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#


from cfg import Config as cfg
import cv2, os, caffe, sys
from other import draw_boxes, resize_im, CaffeModel
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer

NET = "resnet50"
DIR_NAME = 'demo_images'
#DIR_NAME = 'Challenge1_Test_Task12_Images'
#DIR_NAME = 'Challenge2_Test_Task12_Images'
IMAGE_DIR = "test_images/" + DIR_NAME + "/"
NET_DEF_FILE = NET + "/models/deploy.prototxt"
MODEL_FILE = NET + "/train_output/ctpn_" + NET + "_iter_70000.caffemodel"
#MODEL_FILE = NET + "/models/ctpn_trained_model.caffemodel"

caffe.set_mode_cpu()
#caffe.set_device(cfg.GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

demo_imnames=os.listdir(IMAGE_DIR)
demo_imnames = [x for x in demo_imnames if x.split('.')[-1] != 'txt']
timer=Timer()

time_count = 0
count = 0
for im_name in demo_imnames:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%im_name

    im_file=osp.join(IMAGE_DIR, im_name)
    im=cv2.imread(im_file)

    timer.tic()
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im, sq = False)

    print "Number of the detected text lines: %s"%len(text_lines)
    time_now = timer.toc()
    print "Time: %f"%time_now
    time_count += time_now
    count += 1
    print "count:", count
    print "Ave Time:", time_count / count

    im_with_text_lines=draw_boxes(im, text_lines, is_display=False, caption=im_name, wait=False)

    '''
    cv2.imwrite(NET + "/" + DIR_NAME + "_res/" + im_name, im_with_text_lines)
    res_file = open(NET + "/" + DIR_NAME + "_res/res_" + im_name.split('.')[0] + '.txt', 'w')
    for box in text_lines:
        res_file.write(str(int(box[0] / f)) + ',' + str(int(box[1] / f)) + ',' + str(int(box[2] / f)) + ',' + str(int(box[3] / f)) + '\n')
    res_file.close()
    '''
    cv2.imshow("pic", im_with_text_lines)
    cv2.waitKey()


print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."


