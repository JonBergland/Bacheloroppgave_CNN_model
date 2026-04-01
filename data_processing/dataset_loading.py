from torchvision.datasets import ImageFolder


class DatasetLoading():
    """
    A dataclass that loads the dataset from memory and expands it if augmentations are present
    """

    def __init__(self, dataset_root: str):
        self.base_dataset = ImageFolder(root=dataset_root, transform=self.base_transforms)


    ## PSUDO KODE

    ## def getDataset(which data-augmentations):
    ## uses the DataAugmentation class to get the different transforms and then go load 
    ## the dataset with each transform to create a dataset that is a lot larger. 

    