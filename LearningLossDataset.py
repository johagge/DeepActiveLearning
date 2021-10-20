from torch.utils.data import Dataset
import torch.nn.functional as F
from skimage import io
import cv2
import numpy as np
import copy
import torchvision.transforms as transforms
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS


class LearningLossdataset(Dataset):
    '''
    As the learning loss method requires a pytorch dataset, this is just a basic implementation of one.
    This is not useful for any other reasons you would need a pytorch dataset.
    It requires: "dataset - Any pytorch dataset, which has member function "get_image_path"."
    '''
    def __init__(self, images_pool):
        self.images = copy.deepcopy(images_pool)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.get_image_path(index))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_img = transforms.Compose([
            DEFAULT_TRANSFORMS,
            Resize(416)])(
            (img, np.zeros((1, 5))))[0]
        # .unsqueeze(0) removed the unsqueeze, because it's only needed if you only need one image at a time

        # it expects the path to the file and a target, since we have no target, we just add a zero
        # this is not an issue, because the target is discarded anyway in the library
        return input_img, 0



    def get_image_path(self, index):
        return self.images[index]