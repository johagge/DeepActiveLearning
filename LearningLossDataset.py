from torch.utils.data import Dataset

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
        return self.images[index]

    def get_image_path(self, index):
        return self.__getitem__(index)