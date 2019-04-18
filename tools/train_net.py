from cfg import Config as cfg
import caffe
import sys

if len(sys.argv) != 2:
    NET = "resnet50"
else:
    NET = sys.argv[1]

caffe.set_device(cfg.GPU_ID)
caffe.set_mode_gpu()


solver = caffe.SGDSolver(NET + "/models/solver.prototxt")
solver.net.copy_from(NET + "/models/" + NET + ".pretrained.caffemodel")
solver.solve()



