import sys
import math
from pytorchyolo import utils, train
from pytorchyolo import train
import SelectSamples as samples

# hacky way to set arguments...
# this way we can use the argparse used by pytorchyolo
args = [sys.argv[0]]  # put python filename in sys.argv
args += "--model /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/config/robocup.cfg".split(" ")
args += "--data /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/robocup.data".split(" ")
args += "--epochs 5".split(" ")
args += "--pretrained_weights /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/weights/yolov4-tiny.conv.29".split(" ")
args += "--seed 42".split(" ")
args += "--n_cpu 4".split(" ")
args += "--evaluation_interval 1".split(" ")

sys.argv = args  # overwrite sys argv with new arguments

amount = 1000
randomSamples = samples.RandomSampleSelector("/srv/ssd_nvm/15hagge/torso-fuer-pytorchyolo/custom/images/train",
                                             "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/")
imagePoolSize = len(randomSamples.trainImagesPool)

for run in range(math.ceil(imagePoolSize / amount)):
    print(f"___currently running {run} / {range(math.ceil(imagePoolSize / amount))}")
    randomSamples.selectSamples(amount=1000)
    train.run()
    with open("log.txt", "a") as f:
        f.write(f"_done with run: {run}")

