from abc import ABC, abstractmethod
import glob
import os
import random
import shutil
random.seed(42)

class SampleSelector(ABC):
    """Abstract class to select samples"""

    def __init__(self, inputdir, outputdir):
        self.inputdir = inputdir  # directory with images to choose from
        self.outputdir = outputdir  # where list of selected images is located

        self.trainImagesPool = []
        self.findImages(self.inputdir)  # fill list with potential training images
        # images are removed from the pool once they are labeled

        self.trainImages = []  # labeled training images

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

    def __init__(self, inputdir, outputdir):
        super().__init__(inputdir, outputdir)

    def selectSamples(self, amount=100):
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

if __name__ == "__main__":
    a = RandomSampleSelector("/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/custom/images",
                             "/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/data/custom")
    a.selectSamples()
