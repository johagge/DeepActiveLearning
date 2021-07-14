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
args += "--epochs 201".split(" ")  # as the numbers are zero indexed, this will give me evaluation for the 200th run
args += "--pretrained_weights /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/weights/yolov4-tiny.conv.29".split(" ")
args += "--seed 42".split(" ")
args += "--n_cpu 6".split(" ")
args += "--evaluation_interval 10".split(" ")

sys.argv = args  # overwrite sys argv with new arguments

amount = 100

# Generate the first samples randomly assuming we have no suitable heuristic for the first ones
firstSampler = samples.RandomSampleSelector("/srv/ssd_nvm/15hagge/torso-fuer-pytorchyolo/custom/images/train",
                                            "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/")
firstSamples = firstSampler.selectSamples(amount=amount)  # these are used for run 0 of the training

"""
sampler = samples.meanConfidenceSelector("/srv/ssd_nvm/15hagge/torso-fuer-pytorchyolo/custom/images/train",
                                         "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/",
                                         trainImages=firstSamples[0],
                                         trainImagesPool=firstSamples[1],
                                         mode="min")
"""
sampler = samples.RandomSampleSelector("/srv/ssd_nvm/15hagge/torso-fuer-pytorchyolo/custom/images/train",
                                         "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/",
                                         trainImages=firstSamples[0],
                                         trainImagesPool=firstSamples[1])
imagePoolSize = len(sampler.trainImagesPool)


for run in range(10):  # range(math.ceil(imagePoolSize / amount)):
    print(f"___currently running {run} / {range(math.ceil(imagePoolSize / amount))}")
    if run == 0:
        pass  # we already selected samples randomly for this case
    else:
        sampler.selectSamples(amount=amount)
    train.run()
    with open("log.txt", "a") as f:
        f.write(f"_done with run: {run+1}\n")