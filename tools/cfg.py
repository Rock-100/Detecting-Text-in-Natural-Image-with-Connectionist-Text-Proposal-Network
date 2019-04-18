# MUST be imported firstly
import sys
import numpy as np

class Config:
    MEAN=np.float32([102.9801, 115.9465, 122.7717])
    GPU_ID=0
    SCALE=400
    MAX_SCALE=700

    LINE_MIN_SCORE=0.7
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=50
    TEXT_LINE_NMS_THRESH=0.3
    MIN_NUM_PROPOSALS=2
    MIN_RATIO=1.2
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7
    TEXT_PROPOSALS_WIDTH=16

    TRAIN_RPN_POSITIVE_OVERLAP=0.7
    TRAIN_RPN_NEGATIVE_OVERLAP=0.3
    TRAIN_RPN_BATCHSIZE=128
    TRAIN_RPN_FG_FRACTION=0.5
    TRAIN_SIDE_REFINE_MAX=20
    TRAIN_SR_BATCH=64

def init():
    sys.path.insert(0, "./tools")
    sys.path.insert(0, "./caffe/python")
    sys.path.insert(0, "./src")
init()
