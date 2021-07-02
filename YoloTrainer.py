import sys
from pytorchyolo import utils, train
from pytorchyolo import train
import SelectSamples as samples

# hacky way to set arguments...
# this way we can use the argparse used by pytorchyolo
args = [sys.argv[0]]  # put python filename in sys.argv
args += "--model /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/config/robocup.cfg".split(" ")
args += "--data /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/robocup.data".split(" ")
args += "--epochs 25".split(" ")
args += "--pretrained_weights /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/weights/yolov4-tiny.conv.29".split(" ")
args += "--seed 42".split(" ")
args += "--n_cpu 4".split(" ")

print(args)
sys.argv = args  # overwrite sys argv with new arguments

randomSamples = samples.RandomSampleSelector("/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/custom/images",
                                             "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/custom")
randomSamples.selectSamples(amount=1000)
train.run()