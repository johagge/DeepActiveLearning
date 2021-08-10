import sys
import os
from pytorchyolo import train
import SelectSamples as samples
import argparse

import imageDifferenceCalculator

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
cluster_amount = 10
inputdir = "/srv/ssd_nvm/15hagge/torso-fuer-pytorchyolo/custom/images/train"
outputdir = "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/"


# Generate the first samples randomly assuming we have no suitable heuristic for the first ones
firstSampler = samples.RandomSampleSelector(inputdir, outputdir)
firstSamples = firstSampler.selectSamples(amount=amount)  # these are used for run 0 of the training

# Remove cluster from mode, as is it is irrelevant for choice of sampler here
full_mode = trainer_args.mode
if "cluster" in trainer_args.mode:
    trainer_args.mode = trainer_args.mode.replace("_cluster", "")

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
elif trainer_args.mode == "image_2_vec_resnet":
    sampler = samples.DifferenceSampleSelector(inputdir, outputdir, trainImages=firstSamples[0],
                                               trainImagesPool=firstSamples[1], mode="resnet")
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
    elif not "cluster" in full_mode:
        sampler.selectSamples(amount=amount)
    else:  # use clusters for diversity sampling
        # 1. split train pool into clusters
        # 2. for cluster in clusters: use normal sampler
        if not amount % cluster_amount == 0:
            sys.exit("amount of images should be divisible by cluster amount")
        if run == 1:  # we need the train images pool from run 0
            clusterizer = imageDifferenceCalculator.Image2Vector(image_list=firstSamples[1])
        else:
            clusterizer = imageDifferenceCalculator.Image2Vector(image_list=train_images_pool)
        clusterizer.generate_all_image_vectors()
        images_by_cluster = clusterizer.images_by_cluster(cluster_amount)

        # by copying it from the sampler we don't have the if/else we need otherwise for the first run
        previous_train_images_pool = sampler.trainImagesPool.copy()
        # train_images = sampler.trainImages.copy()
        for cluster in images_by_cluster:
            sampler.trainImagesPool = cluster
            _, _, new_train_images = sampler.selectSamples(amount=int(amount/cluster_amount))
            # train_images.extend(new_train_images)

        train_images_pool = set(previous_train_images_pool) - set(sampler.trainImages)
        print(f"There are {len(train_images_pool)} images left in the pool")
        train_images_pool = list(train_images_pool)
        # since we copy the train images pool from the sampler, we also have to set it here
        sampler.trainImagesPool = train_images_pool

        sampler.writeSamplesToFile()


    if len(sampler.trainImages) < (run+1) * amount - 1:  # -1 just in case there is a single one off error
        # TODO verify length of actual train.txt file
        sys.stderr.write("Too few images found")
        sys.stderr.write(f"Found {len(sampler.trainImages)} images in training. Expected {(run+1) * amount}")
        sys.stderr.write(f"currently in run {run}")
        errors.append(f"In run {run} only found {len(sampler.trainImages)} images instead of {(run+1) * amount}")
    train.run()
    with open("log.txt", "a") as f:
        f.write(f"_done with run: {run+1}. Used {(run+1) * amount} images so far.\n")
for error in errors:
    sys.stderr.write(error)