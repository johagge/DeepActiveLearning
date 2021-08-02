import sys
import os
from pytorchyolo import train
import SelectSamples as samples
import argparse

parser = argparse.ArgumentParser(description="Trains a YOLO and selects the most helpful samples for annotating")
parser.add_argument("-m", "--mode", type=str, help="Mode to select samples, e.g. 'random'")

trainer_args = parser.parse_args()



# hacky way to set arguments...
# this way we can use the argparse used by pytorchyolo
args = [sys.argv[0]]  # put python filename in sys.argv
args += "--model /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/config/robocup.cfg".split(" ")
args += "--data /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/robocup.data".split(" ")
args += "--epochs 201".split(" ")  # as the numbers are zero indexed, this provides evaluation results of the 200th run
args += "--pretrained_weights /homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/weights/yolov4-tiny.conv.29".split(" ")
args += "--seed 42".split(" ")
args += "--n_cpu 4".split(" ")
args += "--evaluation_interval 25".split(" ")

sys.argv = args  # overwrite sys argv with new arguments

amount = 100
inputdir = "/srv/ssd_nvm/15hagge/torso-fuer-pytorchyolo/custom/images/train"
outputdir = "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/"


# Generate the first samples randomly assuming we have no suitable heuristic for the first ones
firstSampler = samples.RandomSampleSelector(inputdir, outputdir)
firstSamples = firstSampler.selectSamples(amount=amount)  # these are used for run 0 of the training

if trainer_args.mode == "mean_confidence":
    sampler = samples.meanConfidenceSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                             trainImagesPool=firstSamples[1], mode="mean")
elif trainer_args.mode == "mean_confidence_no_boxes":
    sampler = samples.meanConfidenceSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                             trainImagesPool=firstSamples[1], mode="mean_with_no_boxes")
elif trainer_args.mode == "min_confidence":
    sampler = samples.meanConfidenceSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                             trainImagesPool=firstSamples[1], mode="min")
elif trainer_args.mode == "lowest_max_confidence":
    sampler = samples.meanConfidenceSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                             trainImagesPool=firstSamples[1], mode="lowest_max")
elif trainer_args.mode == "median":
    sampler = samples.meanConfidenceSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                             trainImagesPool=firstSamples[1], mode="median")
elif trainer_args.mode == "max":
    sampler = samples.meanConfidenceSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                             trainImagesPool=firstSamples[1], mode="max")
elif trainer_args.mode == "random":
    sampler = samples.RandomSampleSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                           trainImagesPool=firstSamples[1])
elif trainer_args.mode == "min_bb":
    sampler = samples.BoundingBoxAmountSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                                trainImagesPool=firstSamples[1], mode="least")
elif trainer_args.mode == "most_bb":
    sampler = samples.BoundingBoxAmountSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                                trainImagesPool=firstSamples[1], mode="most")
elif trainer_args.mode == "gaussian_mean_difference":
    sampler = samples.noiseSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                    trainImagesPool=firstSamples[1], mode="gaussian_mean_difference")
elif trainer_args.mode == "gaussian_map_mean":
    sampler = samples.noiseSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                    trainImagesPool=firstSamples[1], mode="gaussian_map_mean")
elif trainer_args.mode == "motion_blur_map_mean":
    sampler = samples.noiseSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                    trainImagesPool=firstSamples[1], mode="motion_blur_map_mean")
else:
    sys.exit("No or incorrect mode was provided")

print(f"Selecting Samples based on {trainer_args.mode}")

if os.path.isfile("log.txt"):
    sys.exit("Logfile already exists.")

with open("log.txt", "a") as f:
    f.write(f"{trainer_args.mode}\n")

imagePoolSize = len(sampler.trainImagesPool)

errors = []

for run in range(10):  # range(math.ceil(imagePoolSize / amount)):
    print(f"___currently running {run} /  9")  # {range(math.ceil(imagePoolSize / amount))}")
    if run == 0:
        pass  # we already selected samples randomly for this case
    else:
        sampler.selectSamples(amount=amount)
    if len(sampler.trainImages) < (run+1) * amount - 1:  # -1 just in case there is a single one off error
        sys.stderr.write("Too few images found")
        sys.stderr.write(f"Found {len(sampler.trainImages)} images in training. Expected {(run+1) * amount}")
        sys.stderr.write(f"currently in run {run}")
        errors.append(f"In run {run} only found {len(sampler.trainImages)} images instead of {(run+1) * amount}")
    train.run()
    with open("log.txt", "a") as f:
        f.write(f"_done with run: {run+1}. Used {(run+1) * amount} images so far.\n")
for error in errors:
    sys.stderr.write(error)