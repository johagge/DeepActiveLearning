import sys
from abc import ABC, abstractmethod
import glob
import os
import random
import shutil
import statistics
from imagecorruptions import corrupt
import cv2
import numpy as np

from tqdm import tqdm
from mean_average_precision import MetricBuilder

import yoloPredictor
import imageDifferenceCalculator


class SampleSelector(ABC):
    """Abstract class to select samples"""

    def __init__(self, inputdir, outputdir, trainImages=None, trainImagesPool=None, seed=42):
        """

        :param inputdir: directory with images to choose from
        :param outputdir: where list of selected images is located
        :param trainImagesPool: if a training pool has already been created, it can be reused here
        """

        random.seed(seed)  # make experiments repeatable
        print(f"using seed {seed}")

        self.inputdir = inputdir
        self.outputdir = outputdir


        if not trainImagesPool:
            self.trainImagesPool = []
            self.findImages(self.inputdir)  # fill list with potential training images
            self.trainImages = []  # labeled training images
        else:
            self.trainImagesPool = trainImagesPool
            self.trainImages = trainImages


    @abstractmethod
    def selectSamples(self, amount=100):
        pass

    def findImages(self, dir):
        datasets = [x[0] for x in os.walk(dir)]  # a list of all subdirectories (including root directory)

        for d in datasets:
            self.trainImagesPool = glob.glob(f"{d}/*.png", recursive=True)
            self.trainImagesPool += glob.glob(f"{d}/*.PNG", recursive=True)
            self.trainImagesPool += glob.glob(f"{d}/*.jpg", recursive=True)
            self.trainImagesPool += glob.glob(f"{d}/*.JPG", recursive=True)

    def writeSamplesToFile(self):
        """"
        writes a list of the used samples to a file
        """
        samples = "\n".join(self.trainImages)
        # with open(os.path.join(self.outputdir, "train.txt", "w")) as f:
        #    f.write(samples)
        with open(os.path.join(self.outputdir, "train.txt"), "w") as f:
            f.write(samples)

    def copyFiles(self, toBeCopied):
        """
        This is currently not used, because we just define it via the train.txt
        Double check if you want to use self.outputdir as the directory to copy images to if you use it
        Copy n files to training folder
        :param toBeCopied: list of files from the pool to be copied into training directory
        :return: None
        """
        for source in toBeCopied:
            # shutil.copy(source, self.outputdir)

            labelFile = source.replace(".png", ".txt").replace(".jpg", ".txt")\
                .replace(".PNG", ".txt").replace(".JPG", ".txt")
            shutil.copy(labelFile, self.outputdir)
            destination = os.path.join(self.outputdir, os.path.basename(source))
            self.trainImages.append(destination)


class RandomSampleSelector(SampleSelector):
    """
    Select samples randomly.
    This is the baseline to compare other approaches to.
    """

    def __init__(self, inputdir, outputdir, trainImages=None, trainImagesPool=None, seed=42):
        super().__init__(inputdir, outputdir, trainImages=trainImages, trainImagesPool=trainImagesPool, seed=seed)

    def selectSamples(self, amount=100):
        """
        selects samples randomly from the pool
        :param amount: amount of images to add to pool
        :return: current train images, pool of remaining images
        """
        # selectedSamples = []
        # for i in range(amount):
            # sample = random.choice(self.trainImagesPool)
            # selectedSamples.append(sample)
            # self.trainImagesPool.remove(sample)
        if amount > len(self.trainImagesPool):  # make sure this doesn't crash at the end
            amount = len(self.trainImagesPool)
        random.shuffle(self.trainImagesPool)
        selectedSamples = self.trainImagesPool[:amount]
        self.trainImages.extend(selectedSamples)
        if amount < len(self.trainImagesPool):
            self.trainImagesPool = self.trainImagesPool[amount:]
        else:
            self.trainImagesPool = [] # there are no images left
        # self.copyFiles(selectedSamples)
        self.writeSamplesToFile()
        return self.trainImages, self.trainImagesPool, selectedSamples

class meanConfidenceSelector(SampleSelector):
    """
    Select the samples which (on average) had the lowest confidences over all predictions
    Questions to evaluate:
        how do we treat no predictions?
            too unsure for a prediction -> should be included?
            no objects -> not important?
            at first ignore them, use them later to prevent false negatives?
    """

    def __init__(self, inputdir, outputdir, trainImages=None, trainImagesPool=None, mode="mean", seed=42):
        super().__init__(inputdir, outputdir, trainImages=trainImages, trainImagesPool=trainImagesPool, seed=seed)
        # we can't load the weights here, because we need new ones after the next training
        self.mode = mode


    def selectSamples(self, amount=100):
        """
        selects samples based on the mean confidences from the pool
        :param amount: amount of images to add to pool
        :return: current train images, pool of remaining images
        """
        if amount > len(self.trainImagesPool):  # make sure this doesn't crash at the end
            amount = len(self.trainImagesPool)

        yolo = yoloPredictor.yoloPredictor()  # load weights here, because after sampling new weights are trained
        predictionConfidences = []

        print("Selecting samples based on confidences:")
        for path in tqdm(self.trainImagesPool):
            boxes = yolo.predict(path)
            # boxes = [[x1, y1, x2, y2, confidence, class]]
            # store one average confidence value per image
            if self.mode == "mean":
                if len(boxes) > 0:
                    confidences = [cfd[4] for cfd in boxes]
                    meanConfidence = statistics.mean(confidences)
                    predictionConfidences.append([meanConfidence, path])

            # prefer images with no bounding boxes, because objects are very common in our domain
            elif self.mode == "mean_with_no_boxes":
                if len(boxes) == 0:
                    meanConfidence = 0
                else:
                    confidences = [cfd[4] for cfd in boxes]
                    meanConfidence = statistics.mean(confidences)
                    predictionConfidences.append([meanConfidence, path])
                predictionConfidences.append([meanConfidence, path])

            elif self.mode == "median":
                if len(boxes) > 0:
                    confidences = [cfd[4] for cfd in boxes]
                    median = statistics.median(confidences)
                    predictionConfidences.append([median, path])

            elif self.mode == "min":
                if len(boxes) > 0:
                    confidences = [cfd[4] for cfd in boxes]
                    minConfidence = min(confidences)
                    predictionConfidences.append([minConfidence, path])

            elif self.mode == "lowest_max" or self.mode == "max":
                if len(boxes) > 0:
                    confidences = [cfd[4] for cfd in boxes]
                    maxConfidence = max(confidences)
                    predictionConfidences.append([maxConfidence, path])


        sortedPredictions = sorted(predictionConfidences)  # sort the list so we can take the first #amount items
        if self.mode == "max":
            # reverse the list, now the highest predictions are at the start of the list
            sortedPredictions = sortedPredictions[::-1]
        new_train_images = []
        for image in sortedPredictions[:amount]:
            self.trainImagesPool.remove(image[1])  # remove about to be labeled images from pool
            self.trainImages.append(image[1])
            new_train_images.append(image[1])
        self.writeSamplesToFile()
        return self.trainImages, self.trainImagesPool, new_train_images


class BoundingBoxAmountSelector(SampleSelector):
    """
    Select the samples which had the most or least bounding box predictions
    """

    def __init__(self, inputdir, outputdir,trainImages=None, trainImagesPool=None, mode="most", seed=42):
        super().__init__(inputdir, outputdir, trainImages=trainImages, trainImagesPool=trainImagesPool,seed=seed)
        # we can't load the weights here, because we need new ones after the next training
        self.mode = mode


    def selectSamples(self, amount=100):
        """
        selects samples based on the amount of predicted bounding boxes from the pool
        :param amount: amount of images to add to pool
        :return: current train images, pool of remaining images
        """
        if amount > len(self.trainImagesPool):  # make sure this doesn't crash at the end
            amount = len(self.trainImagesPool)

        yolo = yoloPredictor.yoloPredictor()  # load weights here, because after sampling new weights are trained
        predictionConfidences = []

        print("Selecting samples based on amount of bounding boxes:")
        for path in tqdm(self.trainImagesPool):
            boxes = yolo.predict(path)

            length = len(boxes)
            predictionConfidences.append([length, path])

        new_train_images = []
        sortedPredictions = sorted(predictionConfidences)  # sort the list so we can take the first #amount items
        if self.mode == "least":
            for image in sortedPredictions[:amount]:
                self.trainImagesPool.remove(image[1])  # remove about to be labeled images from pool
                self.trainImages.append(image[1])
                new_train_images.append(image[1])
        if self.mode == "most":
            for image in sortedPredictions[len(sortedPredictions)-amount:]:
                self.trainImagesPool.remove(image[1])  # remove about to be labeled images from pool
                self.trainImages.append(image[1])
                new_train_images.append(image[1])

        self.writeSamplesToFile()
        return self.trainImages, self.trainImagesPool, new_train_images

class noiseSelector(SampleSelector):
    """"
    Apply noise to the image and compare how much the prediction changes
    The intuition is that a large difference in prediction means uncertain initial prediction
    """

    def __init__(self, inputdir, outputdir,trainImages=None, trainImagesPool=None, mode="", seed=42):
        super().__init__(inputdir, outputdir, trainImages=trainImages, trainImagesPool=trainImagesPool, seed=seed)
        self.mode = mode
        self.number_of_classes = 6

    def selectSamples(self, amount=100):
        """
        selects samples by first applying noise and then comparing the predictions
        :param amount: amount of images to add to pool
        :return: current train images, pool of remaining images
        """
        if amount > len(self.trainImagesPool):  # make sure this doesn't crash at the end
            amount = len(self.trainImagesPool)

        yolo = yoloPredictor.yoloPredictor()  # load weights here, because after sampling new weights are trained


        print("Selecting samples after applying noise to the samples:")
        differences = []
        for path in tqdm(self.trainImagesPool):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # get prediction from uncorrupted image
            init_boxes = yolo.predict(path)

            # apply corruption
            if self.mode == "gaussian_mean_difference" or self.mode == "gaussian_map_mean":
                gaussian_noised_image = corrupt(img, corruption_name="gaussian_noise", severity=1)
                gaussian_boxes = yolo.predictFromLoadedImage(gaussian_noised_image)
            elif self.mode == "motion_mean_difference" or self.mode == "motion_blur_map_mean":
                motion_blurred_image = corrupt(img, corruption_name="motion_blur", severity=3)
                motion_boxes = yolo.predictFromLoadedImage(motion_blurred_image)

            if self.mode == "gaussian_mean_difference":
                difference = self.calc_confidence_difference(init_boxes, gaussian_boxes, "mean")
                differences.append([difference[0], path])
            elif self.mode == "gaussian_map_mean" or self.mode == "motion_blur_map_mean":
                # map library expects for ground truth:
                # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                # pytorchyolo returns
                # # [[x1, y1, x2, y2, confidence, class]]
                # so we need to delete confidence and have class there
                # and set 0, 0 for difficult and crowd
                init_map = []
                for pred in init_boxes:
                    minx, miny, maxx, maxy = self.min_and_max_xy_values(pred)
                    init_map.append([int(minx), int(miny), int(maxx), int(maxy), pred[5], 0, 0])
                init_map = np.array(init_map)
                # map library expects for detection:
                # [xmin, ymin, xmax, ymax, class_id, confidence]
                # -> we need to swap class and confidence
                if self.mode == "gaussian_map_mean":
                    gauss_map = []
                    for pred in gaussian_boxes:
                        minx, miny, maxx, maxy = self.min_and_max_xy_values(pred)
                        gauss_map.append([int(minx), int(miny), int(maxx), int(maxy), pred[5], pred[4]])
                    gauss_map = np.array(gauss_map)
                    noised_map = gauss_map
                if self.mode  == "motion_blur_map_mean":
                    motion_blur_map = []
                    for pred in motion_boxes:
                        minx, miny, maxx, maxy = self.min_and_max_xy_values(pred)
                        motion_blur_map.append([int(minx), int(miny), int(maxx), int(maxy), pred[5], pred[4]])
                    motion_blur_map = np.array(motion_blur_map)
                    noised_map = motion_blur_map

                metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False,
                                                                  num_classes=self.number_of_classes)
                metric_fn.add(noised_map, init_map)
                map_value = metric_fn.value(iou_thresholds=0.5)['mAP']
                differences.append([map_value, path])


        sorted_differences = np.sort(differences, axis=0)  # sort the list so we can take the first #amount items
        print(f"There are {len(sorted_differences)} images in the sorted differences")
        # reverse list so we can just take the first #amount samples
        # this way the images with the biggest difference should be at the start
        if self.mode == "gaussian_mean_difference":
            sorted_differences = sorted_differences[::-1]
        new_train_images = []
        for image in sorted_differences[:amount]:
            self.trainImagesPool.remove(image[1])  # remove about to be labeled images from pool
            self.trainImages.append(image[1])
            new_train_images.append(image[1])
        self.writeSamplesToFile()
        print(f"{len(self.trainImages)} used train images right now. \n"
              f"{len(self.trainImagesPool)} images left in trainImagesPool")
        return self.trainImages, self.trainImagesPool, new_train_images


    def min_and_max_xy_values(self, prediction):
        """
        This is probably unnecessary, but ensure that the order is correct
        :param prediction: yolo prediction
        :return: minx, miny, maxx, maxy
        """
        minx = min(prediction[0], prediction[2])
        maxx = max(prediction[0], prediction[2])
        miny = min(prediction[1], prediction[3])
        maxy = max(prediction[1], prediction[3])
        return minx, miny, maxx, maxy

    def calc_confidence_difference(self, first_preds, second_preds, mode):
        """
        first_preds: initial predictions of neural network with no noise
        second_preds: predictions after noise was applied to the image
        mode: select which values should be compared. options are: mean. maybe in the future: median, max, min
        returns: mean of differences, list of differences
        """
        predictions = [first_preds, second_preds]
        confidences_by_class = []
        for i, pred in enumerate(predictions):
            # add an empty list to the confidences in which we then put the confidences of our predictions
            confidences_by_class.append([])
            # ball, goalpost, robot, L-, T-, X-intersection are the classes in this order
            for object_class in range(0, self.number_of_classes):
                confidences = []
                if len(pred) > 0:
                    for detection in pred:
                        if detection[5] == object_class:
                            confidences.append(detection[4])
                    if mode == "mean":
                        if len(confidences) == 0:
                            confidences_by_class[i].append(0)
                        else:
                            confidences_by_class[i].append(np.mean(confidences))
                else:
                    # this could be any number between the min yolo threshold we use and 0
                    # for simplicity we assume it to be 0
                    confidences_by_class[i].append(0)
        results = []
        for i in range(len(confidences_by_class[0])):
            # calculate difference between e.g. confidence in two ball predictions
            # use absolute numbers, because we don't care which was more confident
            results.append(abs(confidences_by_class[0][i] - confidences_by_class[1][i]))
        return np.mean(results), results

class DifferenceSampleSelector(SampleSelector):
    """
    Select samples by comparing them. No ground truth data is required.
    """

    def __init__(self, inputdir, outputdir, trainImages=None, trainImagesPool=None, mode=None, seed=42):
        super().__init__(inputdir, outputdir, trainImages=trainImages, trainImagesPool=trainImagesPool, seed=seed)
        self.sampler = imageDifferenceCalculator.Image2Vector(trainImagesPool)

    def selectSamples(self, amount=100, cluster_amount=10):
        """
        selects samples from the pool
        :param amount: amount of images to add to pool
        :param cluster_amount: split input images into how many cluster?
        :return: current train images, pool of remaining images
        """
        if amount > len(self.trainImagesPool):  # make sure this doesn't crash at the end
            amount = len(self.trainImagesPool)
        self.sampler.image_list = self.trainImagesPool
        self.sampler.generate_all_image_vectors()
        images_by_cluster = self.sampler.images_by_cluster(cluster_amount)

        current_cluster = 0
        added_images = 0
        print("\n")
        # TODO skip if cluster empty with a warning
        new_train_images = []
        while added_images < amount:
            if len(images_by_cluster[current_cluster]) == 0:
                print(f"No image found for cluster {current_cluster}")
                current_cluster = (current_cluster + 1) % cluster_amount
                continue
            sample = random.choice(images_by_cluster[current_cluster])
            self.trainImages.append(sample)
            new_train_images.append(sample)
            self.trainImagesPool.remove(sample)
            images_by_cluster[current_cluster].remove(sample)

            added_images += 1
            current_cluster = (current_cluster + 1) % cluster_amount
            print(f"Adding image {added_images}/{amount}", end='\r')
        print("\n")

        self.writeSamplesToFile()
        return self.trainImages, self.trainImagesPool, new_train_images



if __name__ == "__main__":
    a = RandomSampleSelector("/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/custom/images",
                             "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/custom")
    a.selectSamples()
