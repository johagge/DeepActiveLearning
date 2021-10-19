from torch.utils.data import Dataset
import torch.nn.functional as F
from skimage import io
import cv2
import numpy as np

class LearningLossdataset(Dataset):
    '''
    As the learning loss method requires a pytorch dataset, this is just a basic implementation of one.
    This is not useful for any other reasons you would need a pytorch dataset.
    It requires: "dataset - Any pytorch dataset, which has member function "get_image_path"."
    '''
    def __init__(self, images_pool):
        self.images = images_pool

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # it expects the path to the file and a target, since we have no target, we just add a zero
        img = io.imread(self.images[index])
        # fix size
        img = cv2.resize(img, (416, 416))
        img = np.rollaxis(img, 2, 0)
        img = img.astype(np.float32) / 255
        return img, 0

    def get_image_path(self, index):
        return self.images[index]